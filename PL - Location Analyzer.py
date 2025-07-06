import pandas as pd
import requests
import json
import os
import zipfile
import xml.etree.ElementTree as ET
import argparse
import sys
from typing import List, Dict, Tuple, Optional
import configparser
import time
from openai import OpenAI
from math import ceil, radians, sin, cos, atan2, sqrt
import logging
from bs4 import BeautifulSoup  # Add BeautifulSoup for HTML parsing
import concurrent.futures
import threading
import re
import traceback
import math # Add math import for distance calculations

# File paths and config
CONFIG_FILE = "config.ini"
DEFAULT_KMZ_FILE = "test123.kmz"
DEFAULT_EXCEL_FILE = "locations.xlsx"
DEFAULT_VERIFICATION_MAP = "verification_map.html"
DEFAULT_CHUNK_SIZE = 10
DEFAULT_API_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_LOCATIONS = 0  # Default maximum number of locations to process
DEFAULT_PAUSE_BEFORE_GPT = False  # Default setting for pausing before GPT estimation
DEFAULT_ENABLE_WEB_BROWSING = False  # Default setting for web browsing
OPENAI_API_ENV_VAR = "OPENAI_API_KEY"  # Environment variable name for API key

class LocationAnalyzer:
    def __init__(self, kmz_file: str, output_excel: str = DEFAULT_EXCEL_FILE, 
                 map_output: str = DEFAULT_VERIFICATION_MAP, verbose: bool = False,
                 openai_api_key: str = "", use_gpt: bool = False, 
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 max_locations: int = DEFAULT_MAX_LOCATIONS,
                 pause_before_gpt: bool = DEFAULT_PAUSE_BEFORE_GPT,
                 enable_web_browsing: bool = DEFAULT_ENABLE_WEB_BROWSING,
                 export_osm_only: bool = False): # <-- Re-added export_osm_only flag
        """Initialize the location analyzer with the KMZ file path.
        
        Args:
            kmz_file: Path to the KMZ file containing the boundary
            output_excel: Path to save the output Excel file
            map_output: Path to save the verification map (not used)
            verbose: Enable verbose output
            openai_api_key: OpenAI API key for GPT population estimation
            use_gpt: Enable GPT population estimation
            chunk_size: Number of locations to process per GPT request
            max_locations: Maximum number of locations to process (0 for no limit)
            pause_before_gpt: Pause before GPT population estimation for review
            enable_web_browsing: Enable web browsing capabilities for GPT (requires GPT-4 and appropriate API access)
            export_osm_only: Export raw collected OSM data (selected tags) and exit immediately
        """
        self.kmz_file = kmz_file
        self.output_excel = output_excel
        self.map_output = map_output
        self.verbose = verbose
        self.openai_api_key = openai_api_key
        self.use_gpt = use_gpt
        self.chunk_size = chunk_size
        self.max_locations = max_locations
        self.pause_before_gpt = pause_before_gpt
        self.enable_web_browsing = enable_web_browsing
        self.export_osm_only = export_osm_only # <-- Re-added storage of the flag
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Initialize OpenAI client if using GPT
        self.gpt_client = None
        if self.use_gpt and self.openai_api_key:
            self.gpt_client = OpenAI(api_key=self.openai_api_key)
            # Determine model based on web browsing setting
            self.gpt_model = "gpt-4-turbo" # Default to a strong model, GPT-4 Turbo often implies browsing capabilities
            print(f"GPT population estimation enabled using model: {self.gpt_model}")
            if not self.enable_web_browsing:
                 print("Warning: Web browsing is disabled in settings, but GPT-4 may still attempt it. Accuracy might be reduced for real-time data.")
            else:
                 print("Web browsing is enabled. GPT will attempt to use real-time data.")
            
        elif self.use_gpt and not self.openai_api_key:
             print("Warning: GPT usage enabled, but no OpenAI API key found. Skipping GPT estimation.")
             self.use_gpt = False # Disable if key is missing
        
        # Load place types from config if available
        self.primary_place_types = []
        self.additional_place_types = []
        self.special_place_types = []
        
        config = read_config()
        if config:
            # Update place types from config
            if "primary_types" in config:
                self.primary_place_types = [t.strip() for t in config["primary_types"].split(",")]
                
            if "additional_types" in config:
                self.additional_place_types = [t.strip() for t in config["additional_types"].split(",")]
                
            if "special_types" in config:
                self.special_place_types = [t.strip() for t in config["special_types"].split(",") if t.strip()]
                
            # Check for OpenAI API key and model
            if "openai_api_key" in config and not self.openai_api_key:
                self.openai_api_key = config["openai_api_key"]
                
            if "gpt_model" in config:
                self.gpt_model = config["gpt_model"]
                
        self.primary_types_pattern = "|".join(self.primary_place_types)
        self.additional_types_pattern = "|".join(self.additional_place_types)
        self.special_types_pattern = "|".join(self.special_place_types) if self.special_place_types else "^$"  # Empty regex that won't match anything
        
        self._print_startup_info(kmz_file, output_excel, max_locations, use_gpt)
        
        # Check if KMZ file exists
        if not os.path.exists(kmz_file):
            raise FileNotFoundError(f"KMZ file not found: {kmz_file}")
    
    def _print_startup_info(
        self, kmz_filepath: str, output_file: str, max_locations: int, use_gpt: bool
    ):
        print(f"Initializing analysis for: {kmz_filepath}")
        print(f"Results will be saved to: {output_file}")
        if max_locations == 0:
            print("No maximum limit on locations")
        else:
            print(f"Maximum locations limit: {max_locations}")
        if use_gpt:
            print("GPT population estimation is enabled")
        else:
            print("GPT population estimation is disabled")
    
    def extract_boundary_from_kmz(self) -> List[Tuple[float, float]]:
        """Extract boundary coordinates from KMZ file."""
        print(f"Extracting boundary from KMZ file: {self.kmz_file}")
        
        try:
            # Open the KMZ file as a zip archive
            with zipfile.ZipFile(self.kmz_file, 'r') as kmz:
                # KMZ files typically contain a doc.kml file
                kml_data = None
                for filename in kmz.namelist():
                    if filename.endswith('.kml'):
                        kml_data = kmz.read(filename)
                        break
                
                if not kml_data:
                    raise ValueError(f"No KML file found in the KMZ archive")
                
                # Parse KML data using ElementTree
                try:
                    root = ET.fromstring(kml_data)
                except ET.ParseError as e:
                    raise ValueError(f"Invalid KML format: {str(e)}")
                
                if self.verbose:
                    print(f"KML root tag: {root.tag}")
                
                # First try to extract namespace from root tag
                namespace = None
                if '{' in root.tag:
                    namespace = root.tag.split('}')[0][1:]
                    if self.verbose:
                        print(f"Detected namespace: {namespace}")
                
                # Use defusedxml to safely parse XML
                def find_coordinates(elem, namespace=None):
                    # Try direct children first
                    if elem.tag.endswith('coordinates') or (namespace and elem.tag == f'{{{namespace}}}coordinates'):
                        return elem.text.strip() if elem.text else None
                    
                    # Try Polygon/LinearRing/coordinates structure
                    for child in elem:
                        if child.tag.endswith('Polygon') or (namespace and child.tag == f'{{{namespace}}}Polygon'):
                            for grand in child:
                                if grand.tag.endswith('outerBoundaryIs') or (namespace and grand.tag == f'{{{namespace}}}outerBoundaryIs'):
                                    for great in grand:
                                        if great.tag.endswith('LinearRing') or (namespace and great.tag == f'{{{namespace}}}LinearRing'):
                                            for great_great in great:
                                                if great_great.tag.endswith('coordinates') or (namespace and great_great.tag == f'{{{namespace}}}coordinates'):
                                                    return great_great.text.strip() if great_great.text else None
                    
                    # Recursively search children
                    for child in elem:
                        result = find_coordinates(child, namespace)
                        if result:
                            return result
                    
                    return None
                
                # Try to find coordinates
                coord_text = find_coordinates(root, namespace)
                
                if not coord_text:
                    # Fallback: Try brute-force search through all elements
                    if self.verbose:
                        print("Using fallback method to find coordinates")
                    
                    for elem in root.findall('.//*'):
                        if elem.tag.endswith('coordinates') or (namespace and elem.tag == f'{{{namespace}}}coordinates'):
                            if elem.text and elem.text.strip():
                                coord_text = elem.text.strip()
                                break
                
                if not coord_text:
                    raise ValueError("No coordinate elements found in KML")
                
                if self.verbose:
                    print(f"Found coordinates: {coord_text[:100]}...")
                
                # Parse coordinates
                geo_points = []
                for coord in coord_text.split():
                    if coord:  # Skip empty strings
                        parts = coord.split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                geo_points.append((lon, lat))
                            except ValueError:
                                continue  # Skip invalid coordinates
                
                if not geo_points:
                    raise ValueError("No valid boundary points found in coordinates")
                
                print(f"Successfully extracted {len(geo_points)} boundary points")
                return geo_points
                
        except Exception as e:
            print(f"Error extracting KMZ file: {str(e)}")
            raise
    
    def is_point_in_polygon(self, point: Tuple[float, float], polygon_points: List[Tuple[float, float]]) -> bool:
        """Check if a point is inside the polygon using ray casting algorithm."""
        x, y = point
        inside = False
        
        j = len(polygon_points) - 1
        for i in range(len(polygon_points)):
            if ((polygon_points[i][1] > y) != (polygon_points[j][1] > y)) and \
               (x < (polygon_points[j][0] - polygon_points[i][0]) * (y - polygon_points[i][1]) / \
                (polygon_points[j][1] - polygon_points[i][1]) + polygon_points[i][0]):
                inside = not inside
            j = i
        
        return inside
    
    # --- Start: Added functions for distance calculation ---
    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points on the Earth."""
        R = 6371  # Earth radius in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2)**2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def point_segment_distance(self, p_lat, p_lon, lat1, lon1, lat2, lon2):
        """Calculate the shortest distance from a point to a line segment using Haversine (approximation)."""
        dist_p1 = self.haversine(p_lat, p_lon, lat1, lon1)
        dist_p2 = self.haversine(p_lat, p_lon, lat2, lon2)
        segment_len = self.haversine(lat1, lon1, lat2, lon2)

        # Handle cases where the segment is very short (effectively a point)
        if segment_len < 1e-9: # Use a small epsilon
            return dist_p1

        # Use simplified projection logic suitable for relatively short segments
        # This calculation assumes a flat plane, which is an approximation for spherical geometry
        dot_product = ((p_lon - lon1) * (lon2 - lon1) + (p_lat - lat1) * (lat2 - lat1))
        squared_len = (lon2 - lon1)**2 + (lat2 - lat1)**2 # Note: Using degrees directly here is an approximation

        t = 0 if squared_len == 0 else max(0, min(1, dot_product / squared_len))

        # Coordinates of the closest point on the infinite line (approximated)
        closest_lon = lon1 + t * (lon2 - lon1)
        closest_lat = lat1 + t * (lat2 - lat1)

        # Calculate distance from the point to this closest point on the line
        dist_to_closest_point = self.haversine(p_lat, p_lon, closest_lat, closest_lon)

        # The actual closest point might be one of the endpoints if the projection falls outside the segment
        if t == 0:
            return dist_p1
        elif t == 1:
            return dist_p2
        else:
            # Otherwise, it's the approximated perpendicular distance to the segment
            return dist_to_closest_point

    def is_point_near_polygon(self, point_lat, point_lon, polygon_points, buffer_km=2.0):
        """Check if a point is within buffer_km of any segment of the polygon boundary."""
        min_dist = float('inf')
        num_points = len(polygon_points)
        if num_points < 2: # Need at least two points to form a segment
            return False

        for i in range(num_points):
            p1_lon, p1_lat = polygon_points[i]
            # Connect the last point back to the first point
            p2_lon, p2_lat = polygon_points[(i + 1) % num_points] 
            
            dist_seg = self.point_segment_distance(point_lat, point_lon, p1_lat, p1_lon, p2_lat, p2_lon)
            min_dist = min(min_dist, dist_seg)
            
            # Optimization: if we find a segment close enough, no need to check further
            if min_dist <= buffer_km:
                return True

        return False # Checked all segments, none were within the buffer
    # --- End: Added functions for distance calculation ---
    
    def _find_osm_locations(self, polygon_points, query_type, place_types=None, skip_city_block=True):
        """Generic function to find OSM locations of various types."""
        min_lat, min_lon, max_lat, max_lon = self._get_bounding_box(polygon_points)
        
        # Build the appropriate query based on query_type
        if query_type == "primary":
            query = self._create_primary_osm_query(min_lat, min_lon, max_lat, max_lon)
        elif query_type == "additional":
            query = self._create_additional_osm_query(min_lat, min_lon, max_lat, max_lon)
        elif query_type == "special":
            query = self._create_special_osm_query(min_lat, min_lon, max_lat, max_lon)
        elif query_type == "administrative":
            query = self._create_administrative_osm_query(min_lat, min_lon, max_lat, max_lon)
        else:
            raise ValueError(f"Invalid query_type: {query_type}")
        
        if self.verbose:
            print(f"OSM Query: {query}")
        
        # Execute the query
        try:
            response = requests.post(self.overpass_url, data=query)
            response.raise_for_status()
            data = response.json()
            
            # Process the results
            return self._process_osm_results(data, place_types, skip_city_block)
        except Exception as e:
            print(f"Error querying OpenStreetMap for {query_type} locations: {str(e)}")
            return []

    # Simplified find_osm_locations function using the generic helper
    def find_osm_locations(self, polygon_points):
        """Find primary location points (cities, towns, etc.) within the polygon."""
        print("Querying OpenStreetMap for primary locations...")
        return self._find_osm_locations(polygon_points, "primary", self.primary_place_types)

    # Simplified find_additional_places function using the generic helper
    def find_additional_places(self, polygon_points):
        """Find additional places (villages, hamlets, suburbs, etc.) within the polygon."""
        print("Querying OpenStreetMap for additional places...")
        return self._find_osm_locations(polygon_points, "additional", self.additional_place_types)

    # Simplified find_special_locations function using the generic helper
    def find_special_locations(self, polygon_points):
        """Find special locations (commercial areas, etc.) within the polygon."""
        if not self.special_place_types:
            print("No special place types defined, skipping special locations search")
            return []
        
        print("Querying OpenStreetMap for special locations...")
        return self._find_osm_locations(polygon_points, "special", self.special_place_types)

    # Simplified find_administrative_areas function using the generic helper
    def find_administrative_areas(self, polygon_points):
        """Find administrative areas (municipalities, districts, etc.) within the polygon."""
        print("Querying OpenStreetMap for administrative areas...")
        return self._find_osm_locations(polygon_points, "administrative")

    def create_verification_map(self, geo_points: List[Tuple[float, float]], locations: List[Dict]):
        """This method is disabled - no HTML map will be generated."""
        # Do nothing - this method is kept for backwards compatibility
        pass
    
    def estimate_populations(self, locations: List[Dict], fetch_hierarchy: bool = True) -> List[Dict]: # Added flag
        """Get population estimates from OSM and GPT (Wikipedia step skipped).""" # Updated docstring
        print(f"\n--- Starting Population Estimation Phase for {len(locations)} locations ---") # Added Start Log
        # print(f"Getting population data for {len(locations)} locations...") # Replaced by above
        
        # --- Fetch Hierarchy (Now Optional) ---
        # ... (hierarchy fetching code remains the same) ...

        # --- Step 1: Fetch OSM Data Only --- 
        print("\n>>> Stage: Fetching population data from OpenStreetMap...") # Enhanced Log
        updated_locations_osm = []
        total_locs_osm = len(locations)
        # Process each location sequentially
        for i, location in enumerate(locations):
            loc_name = location.get('name', f'Location_{i}')
            # Reduced verbosity here, already prints per location
            # print(f"  Processing OSM for location {i+1}/{total_locs_osm}: {loc_name}") 
            
            result = location.copy()  
            result["osm_population"] = None
            
            # Get OSM population
            try:
                result["osm_population"] = self._get_osm_population(location)
                if self.verbose and result["osm_population"] is not None: # Only print if found
                    print(f"  > OSM Pop Found: {loc_name} -> {result['osm_population']}") 
            except Exception as e:
                print(f"Error getting OSM population for {loc_name}: {str(e)}")
            
            # --- Wikipedia Population Step SKIPPED --- 
                
            updated_locations_osm.append(result)
        
        print("<<< Stage: Finished fetching OSM data.") # Enhanced Log

        # --- Step 2: Fetch GPT data in batches (if enabled) ---
        final_locations = updated_locations_osm # Initialize final list
        if self.use_gpt and self.gpt_client:
            print(f"\n>>> Stage: Fetching population data from GPT ({self.gpt_model}) in batches of {self.chunk_size}...") # Enhanced Log
            
            locations_for_gpt = updated_locations_osm 
            
            # Add placeholder columns for GPT results
            for loc in locations_for_gpt:
                loc["gpt_population"] = None
                loc["gpt_confidence"] = None

            num_batches = ceil(len(locations_for_gpt) / self.chunk_size)
            
            for i in range(num_batches):
                start_index = i * self.chunk_size
                end_index = start_index + self.chunk_size
                batch_locations = locations_for_gpt[start_index:end_index]
                
                print(f"  Processing GPT batch {i+1}/{num_batches} (Indices {start_index}-{min(end_index, len(locations_for_gpt))-1})...") # Enhanced Log

                if self.pause_before_gpt and i == 0:
                     input("Press Enter to continue with GPT population estimation...")

                try:
                    gpt_results_map = self._get_gpt_populations_batch(batch_locations, start_index)
                    
                    # Update the main list (locations_for_gpt) with results from this batch
                    for original_idx, result_data in gpt_results_map.items():
                        if 0 <= original_idx < len(locations_for_gpt): 
                            locations_for_gpt[original_idx]["gpt_population"] = result_data.get("population")
                            locations_for_gpt[original_idx]["gpt_confidence"] = result_data.get("confidence")
                            if self.verbose and result_data.get("population") is not None:
                                 print(f"    > GPT Pop Found: {locations_for_gpt[original_idx].get('name')} (Index {original_idx}) -> {result_data.get('population')} ({result_data.get('confidence')})")
                        else:
                            if self.verbose:
                                print(f"Warning: Index {original_idx} from GPT result out of bounds for current list.")
                except Exception as e:
                    print(f"Error processing GPT batch {i+1}: {str(e)}")
                    # Mark locations in this batch as having failed GPT lookup
                    for idx in range(start_index, min(end_index, len(locations_for_gpt))):
                         if 0 <= idx < len(locations_for_gpt):
                            locations_for_gpt[idx]["gpt_confidence"] = "Error"

                print(f"  Finished GPT batch {i+1}/{num_batches}.") # Added batch end log
                time.sleep(1) # Keep delay between batches

            print("<<< Stage: Finished fetching GPT data.") # Enhanced Log
            
            # Save GPT-specific results (using locations_for_gpt)
            print("\nSaving GPT population results to separate file...")
            gpt_results_data = []
            for i, loc in enumerate(locations_for_gpt):
                 admin_hierarchy = loc.get('admin_hierarchy', {}) 
                 gpt_results_data.append({
                     'index': i,
                     'name': loc.get('name'),
                     # 'type': loc.get('type'), # Removed previously
                     # 'latitude': loc.get('latitude'), # Removed previously
                     # 'longitude': loc.get('longitude'), # Removed previously
                     'parent_name': admin_hierarchy.get('parent_name'), 
                     'level_4_name': admin_hierarchy.get('level_4_name'), 
                     'level_2_name': admin_hierarchy.get('level_2_name'), 
                     'gpt_population': loc.get('gpt_population'),
                     'gpt_confidence': loc.get('gpt_confidence')
                 })
            
            # --- Use pd.ExcelWriter for more robust saving --- 
            if gpt_results_data:
                try:
                    gpt_df = pd.DataFrame(gpt_results_data)
                    output_filename = "gpt_population_results.xlsx"
                    # Use ExcelWriter context manager
                    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                        gpt_df.to_excel(writer, index=False, sheet_name='GPT_Results')
                    print(f"GPT results saved to {output_filename}")
                except Exception as e:
                    print(f"Error saving GPT-specific results to Excel using ExcelWriter: {str(e)}")
                    traceback.print_exc() # Print full traceback for errors
            else:
                 print("No GPT results data generated to save.")
            # --- End ExcelWriter usage --- 
            
            final_locations = locations_for_gpt # Update final list reference

        else: # Handle case where GPT is skipped
             print("\n>>> Stage: Skipping GPT population fetch (use_gpt is False or client not initialized).")
             # Ensure columns exist even if GPT is skipped
             for loc in updated_locations_osm:
                loc["gpt_population"] = None
                loc["gpt_confidence"] = None
             final_locations = updated_locations_osm


        # Decide on final population (using final_locations list)
        print("\n>>> Stage: Assigning Final Population (OSM vs GPT)...") # Enhanced Log
        assigned_count = 0
        total_final_locs = len(final_locations)
        for i, loc in enumerate(final_locations):
            osm_pop = loc.get("osm_population")
            gpt_pop = loc.get("gpt_population")
            gpt_conf = loc.get("gpt_confidence")
            loc_name = loc.get("name")
            
            loc["final_population"] = 0 # Default to 0
            loc["population_source"] = "None"
            
            # Logic: Prefer OSM if available, otherwise use GPT if confidence is Medium/High
            if osm_pop is not None and osm_pop > 0:
                loc["final_population"] = osm_pop
                loc["population_source"] = "OSM"
                assigned_count += 1
            elif gpt_pop is not None and gpt_pop > 0 and gpt_conf in ["High", "Medium"]:
                loc["final_population"] = gpt_pop
                loc["population_source"] = "GPT"
                assigned_count += 1
            # else: stays 0 / None
            
            # Optional verbose log for assignment
            # if self.verbose and i % 10 == 0: # Print every 10 locations
            #      print(f"  Assigning final pop... ({i+1}/{total_final_locs})")
        
        print(f"<<< Stage: Finished assigning final population values. Assigned population for {assigned_count}/{total_final_locs} locations.") # Enhanced Log
        
        print("--- Population Estimation Phase Complete ---") # Added End Log

        return final_locations 

    def _get_osm_population(self, location):
        """Get population from OpenStreetMap tags if available."""
        try:
            # OSM data might already be processed if fetched during initial query
            pop_str = location.get("tags", {}).get("population") 
            if pop_str:
                # Try converting to integer, handle potential non-numeric values
                 try:
                     return int(pop_str.replace(',', '').replace(' ', ''))
                 except ValueError:
                     if self.verbose: print(f"Could not parse OSM population '{pop_str}' for {location.get('name')}")
            return None
            return None # Return None if 'population' tag is missing or empty
        except Exception as e:
            if self.verbose:
                print(f"Error extracting OSM population for {location.get('name', 'N/A')}: {str(e)}")
            return None

    def _get_wikipedia_population_data(self, location):
        """Get population from Wikipedia using a multi-stage waterfall extraction strategy."""
        location_name = location['name']
        location_type = location['type']
        admin_hierarchy = location.get('admin_hierarchy', {})
        search_url = "https://en.wikipedia.org/w/api.php"

        if self.verbose:
            print(f"\\n--- Wikipedia Search for: {location_name} ({location_type}) ---")

        # --- Step 1: Define Search Targets (Original + Hierarchy) ---
        search_targets = [{'name': location_name, 'type': location_type, 'level': 'original'}]
        for level in [8, 6, 4]:
            level_name = admin_hierarchy.get(f'level_{level}_name')
            if level_name:
                search_targets.append({'name': level_name, 'type': 'administrative', 'level': f'level_{level}'})

        # --- Step 2: Iterate Through Targets ---
        for target in search_targets:
            target_name = target['name']
            target_type = target['type']
            target_level = target['level']
            if self.verbose: print(f"  Attempting Target: {target_name} (Level: {target_level})")

            # --- Step 3: Define Search Strategies ---
            context_parts_lower = []
            admin_context_str = self._get_administrative_context(location)
            if admin_context_str and not admin_context_str.startswith("Unknown"):
                 parts = admin_context_str.split(',')
                 for part in parts:
                     if ":" in part:
                          value = part.split(':', 1)[1].strip().lower()
                          if value: context_parts_lower.append(value)
            context_query = " ".join(f'"{part}"' for part in context_parts_lower if part not in ['canada', 'usa'])

            search_strategies = []
            if context_query and target_type != 'administrative': search_strategies.append(f'"{target_name}" "{target_type}" {context_query}')
            if context_query: search_strategies.append(f'"{target_name}" {context_query}')
            if target_type != 'administrative': search_strategies.append(f'"{target_name}" "{target_type}"')
            search_strategies.append(f'"{target_name}"')
            search_strategies = [f'{s} population' for s in search_strategies]

            # --- Step 4: Iterate Through Search Strategies ---
            for strategy_query in search_strategies:
                search_results = self._try_wiki_search(search_url, strategy_query)
                if not search_results: continue

                # --- Step 5: Iterate Through Top Search Results ---
                prioritized_result = None
                if target_level != 'original':
                    for res in search_results:
                        res_title_base = res.get("title", "").split(',')[0].strip()
                        if res_title_base.lower() == target_name.lower():
                            prioritized_result = res
                            if self.verbose: print(f"    Prioritizing exact title match: '{res.get('title')}'")
                            break
                
                results_to_process = [prioritized_result] if prioritized_result else search_results[:3] 
                results_to_process = [r for r in results_to_process if r is not None] 

                for result in results_to_process:
                    page_title = result.get("title")
                    snippet = result.get("snippet", "").lower()
                    if not page_title: continue
                    page_title_lower = page_title.lower()

                    if self.verbose: print(f"    Checking result: '{page_title}'")

                    # --- Step 6: Initial Filtering (Snippet & Title) ---
                    name_in_snippet = target_name.lower() in snippet
                    context_in_snippet = any(part in snippet for part in context_parts_lower if len(part) > 3) 
                    name_in_title = target_name.lower() in page_title_lower
                    if not (name_in_title or name_in_snippet):
                        if self.verbose: print(f"      Skipping: Name '{target_name}' not prominent.")
                        continue 
                    if context_parts_lower and not context_in_snippet:
                        if self.verbose: print(f"      Skipping: Context not evident in snippet.")
                        continue
                        
                    # --- Step 7: Fetch Full Page and Validate ---
                    try:
                        content_params = {"action": "parse", "format": "json", "page": page_title, "prop": "text", "formatversion": 2}
                        content_response = requests.get(search_url, params=content_params, timeout=DEFAULT_API_TIMEOUT)
                        content_response.raise_for_status()
                        content_data = content_response.json()
                        html_content = content_data.get("parse", {}).get("text", "")
                        if "error" in content_data or not html_content:
                            if self.verbose: print(f"      Skipping: Error fetching/parsing page '{page_title}'.")
                            continue
            
                        soup = BeautifulSoup(html_content, "html.parser")
                        page_text_lower = soup.get_text().lower()

                        # a) Full Context Check 
                        if context_parts_lower and not any(part in page_text_lower for part in context_parts_lower):
                            if self.verbose: print(f"      Skipping: Full context check failed for '{page_title}'.")
                            continue
                        
                        # b) Granular Target vs. Admin Page Check (simplified)
                        if target_level == 'original' and location_type in self.additional_place_types:
                            name_not_in_title = target_name.lower() not in page_title_lower
                            infobox = soup.find("table", {"class": "infobox"})
                            page_is_clearly_larger_admin_entity = False
                            if infobox:
                                admin_role_keywords = ["city", "town", "municipality", "regional municipality"]
                                if any(kw in infobox.get_text().lower() for kw in admin_role_keywords):
                                    page_is_clearly_larger_admin_entity = True
                            if name_not_in_title and page_is_clearly_larger_admin_entity:
                                if self.verbose: print(f"      Skipping: Page '{page_title}' seems larger admin entity for granular '{location_name}'.")
                                continue

                        # --- Step 8: Waterfall Population Extraction ---
                        if self.verbose: print(f"    Validation passed for '{page_title}'. Starting Population Extraction...")
                        extracted_pop = None
                        extract_source = "None"
                        year_found = 0
                        confidence = 'Low' # Default confidence

                        # Stage 1: Targeted Infobox Extraction
                        infobox_result = self._parse_infobox_population(soup)
                        if infobox_result:
                            extracted_pop = infobox_result['value']
                            year_found = infobox_result['year']
                            extract_source = f"Infobox ({infobox_result['detail']})"
                            confidence = 'High' if year_found >= (time.localtime().tm_year - 5) else 'Medium'
                            if self.verbose: print(f"    DEBUG Waterfall: Stage 1 (Infobox) success. Pop={extracted_pop}, Year={year_found}, Source={extract_source}, Confidence={confidence}")

                        # Stage 2: Lead Sentence Parsing
                        if extracted_pop is None:
                           lead_result = self._parse_lead_sentence_population(soup)
                           if lead_result:
                               extracted_pop = lead_result['value']
                               year_found = lead_result['year']
                               extract_source = "Lead Sentence"
                               confidence = 'Medium' if year_found >= (time.localtime().tm_year - 10) else 'Low' # Adjust confidence based on year
                               if self.verbose: print(f"    DEBUG Waterfall: Stage 2 (Lead) success. Pop={extracted_pop}, Year={year_found}, Confidence={confidence}")

                        # Stage 3: Demographics Section Parsing
                        if extracted_pop is None:
                            demographics_result = self._parse_demographics_population(soup)
                            if demographics_result:
                                extracted_pop = demographics_result['value']
                                year_found = demographics_result['year']
                                extract_source = f"Demographics Section ({demographics_result['detail']})"
                                confidence = 'High' # Census data is good
                                if self.verbose: print(f"    DEBUG Waterfall: Stage 3 (Demographics) success. Pop={extracted_pop}, Year={year_found}, Source={extract_source}, Confidence={confidence}")

                        # Stage 4: Fallback General Text Search
                        if extracted_pop is None:
                            fallback_result = self._parse_fallback_text_population(page_text_lower)
                            if fallback_result:
                                extracted_pop = fallback_result['value']
                                # Year often unreliable here
                                extract_source = f"Fallback Text ({fallback_result['detail']})"
                                confidence = 'Low'
                                if self.verbose: print(f"    DEBUG Waterfall: Stage 4 (Fallback) success. Pop={extracted_pop}, Source={extract_source}, Confidence={confidence}")

                        # --- Step 9: Assign Results and Return ---
                        if extracted_pop is not None:
                            population = extracted_pop
                            notes = f"Success ({target_level} Target): '{target_name}'. Page '{page_title}'. Source: {extract_source}."
                            if year_found > 0: notes += f" Year: {year_found}."
                            notes += f" Strategy: '{strategy_query[:30]}...'"

                            if self.verbose: print(f"    ---> Population Found: {population} ({confidence}) from {extract_source}")

                            if target_level == 'original':
                                return {'pop': population, 'conf': confidence, 'notes': notes}
                            else: # Found pop, but it's for a containing hierarchy level
                                notes = f"Container Found ({extract_source}): Target '{target_name}' ({target_level}). Page '{page_title}'. Pop {population} discarded."
                                if self.verbose: print(f"    ---> Population Found for Container {target_level} '{target_name}'. Discarding.")
                                return {'pop': None, 'conf': None, 'notes': notes}
                        else:
                            if self.verbose: print(f"    Population extraction failed for validated page '{page_title}'.")
                            # Continue to next search result within this strategy

                    except requests.exceptions.RequestException as e:
                        if self.verbose: print(f"    Network error fetching page '{page_title}': {e}")
                    except Exception as e:
                        if self.verbose: print(f"    Error processing page '{page_title}': {e}")
                        traceback.print_exc() # Print full traceback for unexpected parsing errors
                # Finished iterating through results for this strategy
                if self.verbose: print(f"  No valid population found via strategy: {strategy_query[:50]}...")
            # Finished iterating through strategies for this target
            if self.verbose: print(f"  Failed to find population for Target: {target_name}")
        # Finished iterating through all targets
        notes = "Failed to find/validate page or extract population after all targets/strategies."
        if self.verbose: print(f"---> Wikipedia Search Failed for {location_name}")
        return {'pop': None, 'conf': None, 'notes': notes}

    # --- NEW HELPER METHODS FOR WATERFALL EXTRACTION ---

    def _parse_infobox_population(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Stage 1: Extracts population from infobox using targeted logic."""
        infobox = soup.find("table", {"class": "infobox"})
        if not infobox: return None

        possible_pops = []
        # Find all potential population rows (simplistic initial find)
        pop_headers = infobox.find_all("th", string=re.compile(r'Population', re.IGNORECASE))
        
        for header_th in pop_headers:
             year_match = re.search(r'\((\d{4})\)', header_th.get_text())
             year = int(year_match.group(1)) if year_match else 0
             
             # Find the parent row (tr)
             row = header_th.find_parent('tr')
             if not row: continue
             
             # Check for specific labels within this row (or nested)
             # 1. Look for "Total" label
             total_th = row.find("th", string=re.compile(r'^\s*•?\s*Total\s*$', re.IGNORECASE))
             if total_th:
                 cell = total_th.find_next_sibling("td")
                 if cell:
                     value = self._extract_population_number(cell.get_text())
                     if value:
                         possible_pops.append({'value': value, 'year': year, 'detail': 'Total'})
                         continue # Found Total for this year, move to next header
             
             # 2. Look for "City (single-tier)" or "Urban" labels
             city_th = row.find("th", string=re.compile(r'^\s*•?\s*(City|Urban)', re.IGNORECASE))
             if city_th:
                 detail_label = city_th.get_text(strip=True).split('(')[0].strip().replace('•','').strip() # Get 'City' or 'Urban'
                 cell = city_th.find_next_sibling("td")
                 if cell:
                      value = self._extract_population_number(cell.get_text())
                      if value:
                           possible_pops.append({'value': value, 'year': year, 'detail': detail_label})
                           continue # Found City/Urban for this year

             # 3. Look for general population value directly associated with the header
             cell = header_th.find_next_sibling("td")
             if cell:
                 value = self._extract_population_number(cell.get_text())
                 if value:
                     possible_pops.append({'value': value, 'year': year, 'detail': 'Header General'})
        
        if not possible_pops: return None

        # --- Selection Logic ---
        # Prioritize "Total"
        total_candidates = [p for p in possible_pops if p['detail'] == 'Total']
        if total_candidates:
            total_candidates.sort(key=lambda x: x['year'], reverse=True)
            return total_candidates[0]

        # Prioritize "City" or "Urban"
        city_urban_candidates = [p for p in possible_pops if p['detail'] in ['City', 'Urban']]
        if city_urban_candidates:
            # Prefer 'City' over 'Urban' if years are the same or both 0
            city_urban_candidates.sort(key=lambda x: (x['year'], 0 if x['detail'] == 'City' else 1), reverse=True)
            return city_urban_candidates[0]
        
        # Prioritize "Header General" (with year)
        general_candidates = [p for p in possible_pops if p['detail'] == 'Header General']
        if general_candidates:
             general_candidates.sort(key=lambda x: x['year'], reverse=True)
             # Return the one with the latest year, even if it's 0 (if that's all we have)
             return general_candidates[0]
             
        return None # Should not happen if possible_pops was not empty, but safety check

    def _parse_lead_sentence_population(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Stage 2: Extracts population from the lead paragraphs."""
        lead_text = ""
        # Get text from paragraphs before the first main heading (h2)
        for p in soup.find_all('p', recursive=False): # Look for paragraphs directly under main body
            if p.find_previous_sibling('h2'): break # Stop if we hit the first heading
            lead_text += p.get_text() + " "
            if len(lead_text) > 1500: break # Limit search area

        if not lead_text: return None
            
        # Pattern: "... population of approximately [number] residents." (maybe with year)
        # Regex needs refinement - look for "population of X", "population was X", "has X residents/inhabitants" near a year
        patterns = [
             r'population\s+(?:of|was|is|stood at)\s+(?:approximately\s+|about\s+|over\s+)?([\d,]+(?:\.\d+)?)\s*(?:residents|inhabitants|people)?(?:.*?(?:in|as of|census)\s+(\d{4}))?',
             r'(\d{4})\s+(?:census|population)\s+.*?([\d,]+(?:\.\d+)?)\s*(?:residents|inhabitants|people)?',
             r'has\s+([\d,]+(?:\.\d+)?)\s*(?:residents|inhabitants|people)(?:.*?(?:in|as of|census)\s+(\d{4}))?'
        ]

        found_pops = []
        for pattern in patterns:
            for match in re.finditer(pattern, lead_text, re.IGNORECASE):
                 # Extract population and potentially year
                 pop_val = None
                 year_val = 0
                 if len(match.groups()) == 2: # Pattern has pop and year
                     pop_str = match.group(1)
                     year_str = match.group(2)
                     pop_val = self._extract_population_number(pop_str)
                     if year_str: year_val = int(year_str)
                 elif len(match.groups()) == 1: # Pattern likely has only pop
                      pop_str = match.group(1)
                      pop_val = self._extract_population_number(pop_str)
                      # Try to find a year nearby in the sentence (heuristic)
                      sentence_match = re.search(r'(\d{4})', match.group(0))
                      if sentence_match: year_val = int(sentence_match.group(1))
                 
                 if pop_val:
                      found_pops.append({'value': pop_val, 'year': year_val})

        if not found_pops: return None
        
        # Select the best one (latest year)
        found_pops.sort(key=lambda x: x['year'], reverse=True)
        return found_pops[0]

    def _parse_demographics_population(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Stage 3: Extracts population from the 'Demographics' section."""
        demographics_section = None
        # Find h2 or h3 heading with "Demographics"
        for heading in soup.find_all(['h2', 'h3']):
            if heading.get_text(strip=True).lower() == 'demographics':
                demographics_section = heading
                break
        
        if not demographics_section: return None

        section_text = ""
        # Extract text from siblings after the heading until the next heading
        for sibling in demographics_section.find_next_siblings():
            if sibling.name in ['h2', 'h3']: break # Stop at next section
            section_text += sibling.get_text() + " "
            if len(section_text) > 3000: break # Limit search area

        if not section_text: return None

        # Pattern: Focus on census year references
        # E.g., "As of the YYYY census, the population was X" or "YYYY census population of X"
        patterns = [
             r'(\d{4})\s+census.*?(?:population|total)\s+(?:of|was|is)\s+([\d,]+(?:\.\d+)?)',
             r'population\s+(?:of|was|is)\s+([\d,]+(?:\.\d+)?).*?(\d{4})\s+census'
        ]
        found_pops = []
        for pattern in patterns:
             for match in re.finditer(pattern, section_text, re.IGNORECASE):
                  year_val = 0
                  pop_val = None
                  if "census" in match.group(1).lower(): # pop num, year
                     pop_val = self._extract_population_number(match.group(1))
                     year_match = re.search(r'(\d{4})', match.group(2))
                     if year_match: year_val = int(year_match.group(1))
                  else: # year, pop num
                     year_match = re.search(r'(\d{4})', match.group(1))
                     if year_match: year_val = int(year_match.group(1))
                     pop_val = self._extract_population_number(match.group(2))

                  if pop_val and year_val > 0: # Require a year for census data
                      found_pops.append({'value': pop_val, 'year': year_val, 'detail': f'{year_val} Census'})

        if not found_pops: return None
        
        # Select the latest census year
        found_pops.sort(key=lambda x: x['year'], reverse=True)
        return found_pops[0]

    def _parse_fallback_text_population(self, page_text_lower: str) -> Optional[Dict]:
         """Stage 4: Fallback extraction using broad regex on full text."""
         pop_patterns = [
             r'population(?: total)?(?: of| was| is|:)\s*([\\d,]+(?:\\.\\d+)?)',
             r'([\\d,]+(?:\\.\\d+)?)\\s*(?:inhabitants|residents|people)'
         ]
         extracted_pop = None
         pattern_detail = "Unknown"
         for pattern in pop_patterns:
             matches = re.finditer(pattern, page_text_lower, re.IGNORECASE)
             for match in matches:
                 pop_text = match.group(1)
                 found_pop = self._extract_population_number(pop_text)
                 if found_pop is not None:
                     extracted_pop = found_pop
                     pattern_detail = f"Pattern: {pattern[:20]}..."
                     break # Take first match from this pattern
             if extracted_pop is not None: break # Take first successful pattern

         if extracted_pop:
             return {'value': extracted_pop, 'detail': pattern_detail}
         return None

    # --- END NEW HELPER METHODS ---

    def _try_wiki_search(self, search_url: str, query: str) -> Optional[List[Dict]]:
        """Helper function to perform a single Wikipedia search query, returns results with snippets."""
        search_params = {
            "action": "query", "format": "json", "list": "search",
            "srsearch": query, 
            "srlimit": 5, # Limit results 
            "srprop": "snippet" # Request snippet
        }
        try:
            if self.verbose: print(f"  Trying Wiki search: {query}")
            search_response = requests.get(search_url, params=search_params, timeout=DEFAULT_API_TIMEOUT)
            search_response.raise_for_status() 
            search_data = search_response.json()
            results = search_data.get("query", {}).get("search")
            if results:
                 if self.verbose: print(f"    Found {len(results)} results.")
                 return results
            else:
                 if self.verbose: print(f"    No results found.")
                 return None
        except requests.exceptions.RequestException as e:
            if self.verbose: print(f"    Network error during Wiki search: {e}")
            return None
        except Exception as e:
             if self.verbose: print(f"    Error during Wiki search: {e}")
             # Correct indentation for this return statement
             return None

    def _extract_population_number(self, text: str) -> Optional[int]:
        """Helper to extract the first valid integer population number from text."""
        # Remove commas, spaces, and anything after a parenthesis or bracket (like citations)
        text = re.split(r'[\(\[]', text)[0] # Take text before first ( or [
        cleaned_text = ''.join(filter(str.isdigit, text))
        
        if cleaned_text:
            try:
                pop_int = int(cleaned_text)
                # Add basic sanity check (e.g., ignore very small or extremely large numbers)
                if 10 <= pop_int <= 500000000: # Population between 10 and 500 million
                    return pop_int
            except ValueError:
                pass # Not a valid integer
        # Correct indentation for final return statement
        return None

    def _prepare_batch_gpt_prompt(self, locations_chunk: List[Dict], start_index: int) -> str:
        """Creates the specific prompt for a batch of locations for GPT population estimation,
        requesting a JSON array output.
        
        Args:
            locations_chunk: List of location dictionaries for the current batch.
            start_index: The starting index of this chunk in the original list.

        Returns:
            The formatted prompt string.
        """
        prompt_header = ( 
            "You are an AI assistant specializing in accurately retrieving demographic data based on geographic context. "
            "Your task is to estimate the most recent known residential population for each of the locations listed below.\n\n"
            "For each location, you are provided with:\n"
            "- 'index': An identifier for the location within the overall list (starting from {start_index}).\n" # Clarified index meaning
            "- 'name': The name of the specific location.\n"
            "- 'type': The type of the location (e.g., city, suburb, neighbourhood).\n"
            "- 'parent_name': The name of the administrative area directly containing the location.\n"
            "- 'level_4_name': The name of the higher-level administrative area (e.g., province or state).\n"
            "- 'level_2_name': The name of the country.\n\n"
            "**Crucially, use the provided 'type', 'parent_name', 'level_4_name', and 'level_2_name' to disambiguate the location 'name' "
            "and ensure you are retrieving the population for the correct place.**\n\n"
            # --- Requesting JSON Array --- 
            "Provide your answer ONLY as a single, valid JSON **array**. Each object in the array must correspond to one location from the input list below, respecting the original order.\n"
            "Do not include any introductory text, explanations, or summaries before or after the JSON array.\n"
            "Each object in the JSON array must contain:\n"
            "- 'index': The original integer index provided for the location.\n"
            "- 'population': The estimated population as an integer. If the population is unknown or cannot be reliably estimated (see instruction below), use `null`.\n"
            "- 'confidence': Your confidence in the estimate as a string: 'High', 'Medium', or 'Low'.\n\n"
            # --- End JSON Array Request ---
            "**Estimation Guidance:** For locations with type 'suburb' or 'neighbourhood', if a direct, reliable population figure isn't found in your knowledge, "
            "provide a reasonable **estimate** based on the context (parent_name, level_4_name) and typical population densities for such areas. "
            "Assign **'Medium'** confidence to these well-informed estimates. Assign 'Low' confidence only if even estimation is highly uncertain.\n\n"
            # --- Updated Example Format (JSON Array) --- 
            "Example Response Format (for a batch of 3 locations with indices 0, 1, 2):\n"
            "[\n"
            "  {\"index\": 0, \"population\": 186948, \"confidence\": \"High\"},\n"
            "  {\"index\": 1, \"population\": 25000, \"confidence\": \"Medium\"}, \"# Example estimate\"\n"
            "  {\"index\": 2, \"population\": null, \"confidence\": \"Low\"}\n"
            "]\n\n"
             # --- End Updated Example ---
            "--- START LOCATIONS ---"
        )
        
        locations_data = []
        for i, loc in enumerate(locations_chunk):
            original_index = start_index + i
            admin_hierarchy = loc.get('admin_hierarchy', {}) 
            
            location_info = {
                "index": original_index, # Use the global index directly
                "name": loc.get('name', 'Unknown'),
                "type": loc.get('type', 'unknown'), 
                "parent_name": admin_hierarchy.get('parent_name'), 
                "level_4_name": admin_hierarchy.get('level_4_name'),
                "level_2_name": admin_hierarchy.get('level_2_name')
            }
            locations_data.append(location_info)
            
        locations_json_str = json.dumps(locations_data, indent=2, ensure_ascii=False)

        prompt_footer = (
            "\n--- END LOCATIONS ---\n\n"
            "Provide ONLY the JSON array containing one object for each location listed above, matching the provided indices."
        )

        final_prompt = f"{prompt_header}\n{locations_json_str}\n{prompt_footer}"
        
        if start_index == 0 and self.verbose:
             print("\n--- Sample GPT Prompt (Batch 0) ---")
             print(final_prompt)
             print("------------------------------------\n")
             
        return final_prompt

    def _get_gpt_populations_batch(self, locations_chunk: List[Dict], start_index: int) -> Dict[int, Dict]:
        """Sends a batch of locations to GPT and parses the JSON array response.

        Args:
            locations_chunk: List of location dictionaries for the batch.
            start_index: The starting global index for this batch.

        Returns:
            A dictionary mapping the original global index to the parsed result 
            (e.g., {10: {'population': 16500, 'confidence': 'Medium'}}).
        """
        prompt = self._prepare_batch_gpt_prompt(locations_chunk, start_index)
        
        # --- API Call --- 
        try:
            response = self.gpt_client.chat.completions.create(
                model=self.gpt_model, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, 
                # REMOVED response_format={\"type\": \"json_object\"}
            )
            raw_response_content = response.choices[0].message.content
            if self.verbose:
                 print("\n============================== GPT Raw Response ==============================")
                 print(raw_response_content)
                 print("================================================================================\n")
                 
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            results_map = {}
            for i in range(len(locations_chunk)):
                original_idx = start_index + i
                results_map[original_idx] = {"population": None, "confidence": "API Error"}
            return results_map

        # --- Parse the Response (Expecting JSON ARRAY now) --- 
        parsed_results_map = {}
        try:
            # Attempt to parse the entire response as JSON (expecting a list)
            gpt_response_json = json.loads(raw_response_content)
            
            # Check if it is a list (JSON array)
            if not isinstance(gpt_response_json, list):
                print(f"Error: GPT response was not a JSON array/list as expected. Type: {type(gpt_response_json)}")
                # Attempt to extract array with regex as a fallback
                match = re.search(r'\[\s*\{.*?\}\s*\]', raw_response_content, re.DOTALL)
                if match:
                    json_text = match.group(0)
                    try:
                        gpt_response_json = json.loads(json_text)
                        if not isinstance(gpt_response_json, list):
                             raise ValueError("Regex extracted content is not a list.")
                        if self.verbose: print("Successfully parsed JSON array extracted via regex.")
                    except Exception as e_regex:
                        print(f"Error: Failed to parse JSON array even after regex extraction: {e_regex}")
                        raise ValueError("Response is not a valid JSON array, even after regex.")
                else:
                    raise ValueError("Response is not a JSON array and regex extraction failed.")

            # Iterate through the items (dictionaries) in the list
            processed_indices = set()
            for item_data in gpt_response_json:
                try:
                    if not isinstance(item_data, dict):
                        print(f"Warning: Item in response array is not a dictionary: {item_data}. Skipping.")
                        continue

                    original_idx = item_data.get('index')
                    population = item_data.get('population')
                    confidence = item_data.get('confidence')

                    # Validate index is an integer and within expected range
                    if not isinstance(original_idx, int): 
                        print(f"Warning: Index '{original_idx}' is not an integer. Skipping item: {item_data}")
                        continue
                         
                    if not (start_index <= original_idx < start_index + len(locations_chunk)):
                        print(f"Warning: Index {original_idx} from GPT response is outside the expected range for this batch ({start_index} to {start_index + len(locations_chunk) - 1}). Skipping item: {item_data}")
                        continue
                    
                    # Validate population
                    pop_value = None
                    if population is not None:
                        try:
                            pop_float = float(population)
                            if pd.notna(pop_float):
                                pop_value = int(pop_float) 
                        except (ValueError, TypeError):
                             if self.verbose:
                                print(f"Warning: Could not parse population '{population}' for index {original_idx}. Setting to None.")
                            
                    # Validate confidence
                    conf_value = str(confidence) if confidence in ["High", "Medium", "Low"] else None
                    if not conf_value and self.verbose and confidence is not None:
                         print(f"Warning: Invalid confidence '{confidence}' for index {original_idx}. Setting to None.")

                    # Store parsed data, preventing duplicates
                    if original_idx in processed_indices:
                         print(f"Warning: Duplicate index {original_idx} found in GPT response. Skipping.")
                         continue
                    parsed_results_map[original_idx] = {
                        "population": pop_value, 
                        "confidence": conf_value
                    }
                    processed_indices.add(original_idx)
                
                except Exception as e: # Correctly indented under the `for item_data` loop
                     print(f"Warning: Unexpected error processing item in array: {item_data}. Error: {e}. Skipping.")

            # Check if we got results for all expected indices (This block is outside the inner `for` loop)
            if len(parsed_results_map) != len(locations_chunk):
                print(f"Warning: Number of valid results ({len(parsed_results_map)}) does not match batch size ({len(locations_chunk)}).")
                 
            for i in range(len(locations_chunk)):
                expected_idx = start_index + i
                if expected_idx not in parsed_results_map:
                    print(f"Warning: No valid result found for expected index {expected_idx}.")
                    parsed_results_map[expected_idx] = {"population": None, "confidence": "Missing"}

        except json.JSONDecodeError as e: # Correctly aligned with the outer `try`
            print(f"Error: Failed to decode GPT JSON response: {str(e)}")
            for i in range(len(locations_chunk)):
                 original_idx = start_index + i
                 parsed_results_map[original_idx] = {"population": None, "confidence": "Parse Error"}
        except ValueError as e: # Correctly aligned with the outer `try`
             print(f"Error processing GPT response structure: {e}")
             for i in range(len(locations_chunk)):
                 original_idx = start_index + i
                 parsed_results_map[original_idx] = {"population": None, "confidence": "Format Error"}
        except Exception as e: # Correctly aligned with the outer `try`
            print(f"Error during GPT response parsing: {str(e)}")
            traceback.print_exc() 
            for i in range(len(locations_chunk)):
                 original_idx = start_index + i
                 parsed_results_map[original_idx] = {"population": None, "confidence": "Parse Error"}

        return parsed_results_map 

    def _get_administrative_context(self, location):
        """Get the administrative context (county/region, state/province, country) for a location using OSM."""
        lat = location.get('latitude')
        lon = location.get('longitude')
        if lat is None or lon is None:
            return "Unknown (Missing Coordinates)"
            
        # Query OSM for administrative boundaries containing the point
        # We query multiple levels (e.g., 2=country, 4=state/province, 6=region/county, 8=municipality)
        query = f"""
        [out:json][timeout:30];
        is_in({lat},{lon}) -> .a;
        (
          area.a[admin_level="2"]; // Country
          area.a[admin_level="4"]; // State/Province
          area.a[admin_level="6"]; // Region/County
          area.a[admin_level="8"]; // Municipality/City District
        );
        out tags;
        """
        
        context_parts = {"country": None, "province": None, "county": None, "municipality": None}
        
        try:
            response = requests.post(self.overpass_url, data=query, timeout=DEFAULT_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                level = tags.get('admin_level')
                name = tags.get('name') or tags.get('name:en') # Prefer English name if available
                
                if not name: continue # Skip if no name found

                if level == '2':
                    context_parts["country"] = name
                elif level == '4':
                    context_parts["province"] = name
                elif level == '6':
                    context_parts["county"] = name
                elif level == '8':
                     context_parts["municipality"] = name

            # Construct the context string, prioritizing more specific levels
            # Focus on Level 4 (State/Province) and Level 6 (County/Region) as requested indirectly
            final_context = []
            if context_parts["county"]: final_context.append(f"County/Region: {context_parts['county']}")
            if context_parts["province"]: final_context.append(f"State/Province: {context_parts['province']}")
            if context_parts["country"]: final_context.append(f"Country: {context_parts['country']}")
            
            if final_context:
                return ", ".join(final_context)
            elif context_parts["municipality"]: # Fallback if others are missing
                 return f"Municipality: {context_parts['municipality']}" + (f", {context_parts['country']}" if context_parts["country"] else "")
            else:
                 return "Unknown administrative context" # Fallback if query fails or returns nothing useful
            
        except requests.exceptions.RequestException as e:
            if self.verbose: print(f"Network error getting administrative context for {location.get('name', 'N/A')}: {str(e)}")
            return "Unknown (Network Error)"
        except Exception as e:
            if self.verbose: print(f"Error parsing administrative context for {location.get('name', 'N/A')}: {str(e)}")
            return "Unknown (Processing Error)"

    def save_to_excel(self, all_locations=None, filename=None):
        """Save final analysis results to Excel with specified columns only."""
        if all_locations is None or not all_locations: # Check if list is empty
            print("No locations to save.")
            return False
        
        if not filename:
            filename = self.output_excel
        
        # --- DEBUG PRINT 1: Show first 5 items received by the function ---
        print("\\n--- DEBUG: Data received by save_to_excel (first 5) ---")
        for i, loc in enumerate(all_locations[:5]):
             print(f"  Item {i}: Name={loc.get('name')}, WikiPop={loc.get('wiki_population')}, GPTPop={loc.get('gpt_population')}")
        print("-------------------------------------------------------\\n")
        # --- END DEBUG ---
        
        try:
            # Convert to DataFrame, add placeholder for admin_hierarchy if missing
            for loc in all_locations:
                 if 'admin_hierarchy' not in loc: loc['admin_hierarchy'] = {}
                 
            df = pd.json_normalize(all_locations, sep='_') # Normalize nested admin_hierarchy
            
            # Define the exact columns we want in the exact order
            final_columns = [
                'name',
                'type',
                'latitude',
                'longitude',
                'admin_hierarchy_containing_level',
                'admin_hierarchy_parent_name',
                'admin_hierarchy_level_4_name',
                'osm_population_tag',
                'osm_population_date',
                'gpt_population',
                'gpt_confidence'
            ]
            
            # Ensure all expected columns exist, adding missing ones with None
            for col in final_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Select and reorder final columns
            df = df[final_columns]
            
            # Clean up column names by removing the 'admin_hierarchy_' prefix
            df.columns = [col.replace('admin_hierarchy_', '') for col in df.columns]
            
            # Convert numeric columns
            numeric_cols = ['osm_population_tag', 'gpt_population', 'containing_level']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Float64')

            # --- DEBUG PRINT 2: Show head of DataFrame before saving ---
            print("\\n--- DEBUG: DataFrame head before saving to Excel ---")
            print(df.head())
            print("----------------------------------------------------\\n")
            # --- END DEBUG ---
            
            # Save to Excel
            df.to_excel(filename, index=False, engine='openpyxl')
            print(f"\\nSaved {len(all_locations)} locations to Excel file: {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving to Excel: {str(e)}")
            traceback.print_exc() # Print stack trace for debugging Excel errors
            return False
    
    def run(self):
        """Run the full analysis process, with option for OSM-only export."""
        try:
            self.polygon_points = self.extract_boundary_from_kmz()
            
            # --- Find OSM Locations (Includes extra tags) ---
            print("\n--- Finding OSM Locations ---")
            primary_locations = self._find_osm_locations(self.polygon_points, "primary", self.primary_place_types)
            
            # Only query additional places if we have additional types configured
            additional_places = []
            if self.additional_place_types:
                additional_places = self._find_osm_locations(self.polygon_points, "additional", self.additional_place_types)
            
            # Only query special locations if we have special types configured
            special_locations = []
            if self.special_place_types: 
                special_locations = self._find_osm_locations(self.polygon_points, "special", self.special_place_types)
            
            administrative_areas = [] 
            
            # Combine and deduplicate
            all_locations = primary_locations + additional_places + special_locations + administrative_areas
            all_locations = self.clean_and_deduplicate_locations(all_locations)
            
            # --- Fetch Administrative Hierarchy *Immediately* After OSM Search --- 
            print("\n--- Fetching Administrative Hierarchy ---")
            locations_with_hierarchy = []
            total_locs = len(all_locations)
            for i, location in enumerate(all_locations):
                if self.verbose: print(f"  Fetching hierarchy for {location['name']} ({i+1}/{total_locs})...")
                # Fetch hierarchy and update the location dictionary directly
                location['admin_hierarchy'] = self._fetch_admin_hierarchy(location) 
                locations_with_hierarchy.append(location) # Keep track, although modifying in place
                # Optional: Add a small delay if needed, but might slow down export mode
                # time.sleep(0.1) 
            all_locations = locations_with_hierarchy # Ensure we use the list with updated dicts
            print("Finished fetching hierarchy.")
            # --- End Immediate Hierarchy Fetch --- 

            print("\n--- OSM Location Summary (Post-Hierarchy) ---")
            print(f"Total unique OSM locations found: {len(all_locations)}")

            # --- Check for OSM Export Only Flag --- 
            if self.export_osm_only:
                print("\n--export-osm-only flag detected. Exporting collected OSM data and hierarchy, then exiting.")
                export_filename = "osm_data_export.xlsx"
                # Call the dedicated OSM export function (which now needs hierarchy columns)
                success = self.save_osm_data_to_excel_only(all_locations, export_filename) 
                if success:
                    print(f"OSM data saved to {export_filename}")
                else:
                    print(f"Failed to save OSM data to {export_filename}")
                return success # Stop execution here
            # --- End OSM Export Check ---

            # --- Limit Locations if Needed (Apply *after* potential OSM export) ---
            if self.max_locations > 0 and len(all_locations) > self.max_locations:
                print(f"\nLimiting results to {self.max_locations} locations (out of {len(all_locations)} found)")
                # Simplified prioritization: primary > additional > special > admin
                prioritized = []
                prioritized.extend(primary_locations)
                remaining = self.max_locations - len(prioritized)
                if remaining > 0: prioritized.extend(additional_places[:remaining])
                remaining = self.max_locations - len(prioritized) # Corrected indentation
                if remaining > 0: prioritized.extend(special_locations[:remaining])
                remaining = self.max_locations - len(prioritized) # Corrected indentation
                if remaining > 0: prioritized.extend(administrative_areas[:remaining])
                all_locations = prioritized[:self.max_locations]
                print(f"Processing {len(all_locations)} locations after limiting.")

            # --- Proceed with Full Analysis (Hierarchy is already fetched) ---
            print("\n--- Starting Population Estimation (Wiki/GPT) ---")
            # estimate_populations no longer needs to fetch hierarchy itself
            all_locations = self.estimate_populations(all_locations, fetch_hierarchy=False) # Add flag to skip fetch inside
            
            # Save final combined results
            self.save_to_excel(all_locations)
            
            # ... (Final summary print statements) ...
            return True
            
        except Exception as e:
            print(f"An error occurred during analysis: {str(e)}")
            traceback.print_exc()
            return False

    def clean_and_deduplicate_locations(self, locations: List[Dict]) -> List[Dict]:
        """
        Clean location names by replacing hyphens with spaces and remove duplicates.
        
        Args:
            locations: List of location dictionaries
            
        Returns:
            List of cleaned and deduplicated locations
        """
        # Clean location names (replace hyphens with spaces)
        for location in locations:
            if 'name' in location:
                location['name'] = location['name'].replace('-', ' ')
                
        # Remove duplicates based on name
        seen_names = set()
        unique_locations = []
        
        for location in locations:
            location_name = location.get('name', '')
            if location_name and location_name not in seen_names:
                seen_names.add(location_name)
                unique_locations.append(location)
            elif self.verbose and location_name:
                print(f"Removing duplicate location: {location_name}")
                
        return unique_locations

    def _get_bounding_box(self, geo_points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Get the bounding box of a polygon with buffer."""
        lons, lats = zip(*geo_points)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Small buffer to avoid edge cases
        buffer = 0.04
        min_lon -= buffer
        max_lon += buffer
        min_lat -= buffer
        max_lat += buffer
        
        if self.verbose:
            print(f"Bounding box: lat {min_lat} to {max_lat}, lon {min_lon} to {max_lon}")
        
        return min_lat, min_lon, max_lat, max_lon

    def _create_primary_osm_query(self, min_lat, min_lon, max_lat, max_lon):
        """Create query for primary locations."""
        return self._create_osm_query(min_lat, min_lon, max_lat, max_lon, self.primary_types_pattern)

    def _create_additional_osm_query(self, min_lat, min_lon, max_lat, max_lon):
        """Create query for additional places."""
        return self._create_osm_query(min_lat, min_lon, max_lat, max_lon, self.additional_types_pattern)

    def _create_special_osm_query(self, min_lat, min_lon, max_lat, max_lon):
        """Create query for special locations (commercial areas)."""
        return f"""
        [out:json][timeout:60];
        (
          node["landuse"="commercial"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["landuse"="commercial"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["landuse"="commercial"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

    def _create_administrative_osm_query(self, min_lat, min_lon, max_lat, max_lon):
        """Create query for administrative areas."""
        return f"""
        [out:json][timeout:60];
        (
          relation["boundary"="administrative"]["admin_level"~"2|3|4|5|6"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

    def _create_osm_query(self, min_lat, min_lon, max_lat, max_lon, place_types_pattern):
        """Create a common OSM Overpass query for different place types."""
        return f"""
        [out:json][timeout:60];
        (
          node["place"~"{place_types_pattern}"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["place"~"{place_types_pattern}"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["place"~"{place_types_pattern}"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

    def _process_osm_results(self, response_data, place_type_filter=None, skip_city_block=True):
        """Process OSM query results, store selected tags, and filter by geometry."""
        locations = []
        if "elements" not in response_data:
            return locations
        
        osm_elements = response_data["elements"]
        print(f"Processing {len(osm_elements)} OSM elements...") # Debug print
        
        for element in osm_elements:
            tags = element.get("tags", {})
            element_type = element.get("type")
            element_id = element.get("id")

            # --- Determine Type and Name ---
            location_type = None
            if "place" in tags:
                place_type = tags["place"]
                # Skip if we're filtering by place type and this doesn't match
                if place_type_filter and place_type not in place_type_filter:
                    continue
                # Skip city_block if needed
                if skip_city_block and place_type == "city_block":
                    continue
                location_type = place_type
            elif tags.get("landuse") == "commercial":
                 # Allow processing commercial areas if place_type_filter is None or matches special types
                 if place_type_filter is None or "commercial_area" in place_type_filter: 
                     location_type = "commercial_area"
                 else:
                     continue # Skip if we are filtering and it doesn't match
            elif tags.get("boundary") == "administrative":
                # Allow processing admin areas if place_type_filter is None (usually queried separately)
                if place_type_filter is None: 
                    location_type = "administrative"
                else:
                    continue # Skip if we are filtering and it doesn't match
            else:
                continue # Skip elements without relevant type tags

            name = tags.get("name", "")
            if not name:
                if self.verbose: print(f"Skipping element {element_id} ({location_type}) due to missing name.")
                continue
                    
            # --- Get Coordinates ---
            lat, lon = 0, 0
            if element_type == "node":
                lat, lon = element.get("lat", 0), element.get("lon", 0)
            elif "center" in element:
                lat, lon = element["center"].get("lat", 0), element["center"].get("lon", 0)
            elif "lat" in element and "lon" in element: # Sometimes center might be missing?
                lat, lon = element.get("lat", 0), element.get("lon", 0)
            else:
                if self.verbose: print(f"Skipping {name} ({location_type}) due to missing coordinates.")
                continue
                
            # --- Geometry Check (Inside or Near Polygon) ---
            is_inside = self.is_point_in_polygon((lon, lat), self.polygon_points)
            is_near = False
            if not is_inside:
                is_near = self.is_point_near_polygon(lat, lon, self.polygon_points, buffer_km=2.0)

            if not (is_inside or is_near): # Restore the check for both inside OR near
                continue # Skip if not inside or near

            # --- Extract Desired Tags (Modified) --- 
            osm_population_tag = tags.get('population', None)
            try:
                if osm_population_tag is not None:
                     osm_population_tag = int(str(osm_population_tag).replace(',','').replace(' ',''))
                else:
                     osm_population_tag = None
            except (ValueError, TypeError):
                 if self.verbose: print(f"Warning: Could not parse population tag '{tags.get('population')}' for {name}")
                 osm_population_tag = None
                 
            osm_population_date = tags.get('population:date', None)
            wikidata_id = tags.get('wikidata', None)
            wikipedia_link = tags.get('wikipedia', None)
            # Removed osm_is_in and osm_admin_level extraction
            
            # --- Append Location Data (Modified) ---    
            location_data = {
                        "name": name,
                "type": location_type,
                # "source": "OpenStreetMap", # Removed source
                        "latitude": lat,
                "longitude": lon,
                "osm_population_tag": osm_population_tag,
                "osm_population_date": osm_population_date,
                "wikidata_id": wikidata_id,
                "wikipedia_link": wikipedia_link,
                # Removed osm_is_in, osm_admin_level
            }
            locations.append(location_data)
            
            if self.verbose: 
                inclusion_reason = "inside boundary" if is_inside else "near boundary (within 2km)"
                print(f"OSM Add: {name} ({location_type}) -> Pop: {osm_population_tag}, Date: {osm_population_date}, Wiki: {wikipedia_link} ({inclusion_reason})")
        
        print(f"Finished processing OSM elements. Added {len(locations)} locations.") # Debug print
        return locations

    # --- NEW Function to Fetch Hierarchy ---
    def _fetch_admin_hierarchy(self, location):
        """Get the administrative hierarchy (levels 8, 6, 4, 2) and determine containing/parent levels."""
        lat = location.get('latitude')
        lon = location.get('longitude')
        # Initialize hierarchy dict with names and the new level fields
        hierarchy = {f'level_{level}_name': None for level in [8, 6, 4, 2]}
        hierarchy['containing_level'] = None
        hierarchy['parent_level'] = None
        hierarchy['parent_name'] = None

        if lat is None or lon is None:
            return hierarchy # Return empty if no coordinates

        query = f"""
        [out:json][timeout:{DEFAULT_API_TIMEOUT}];
        is_in({lat},{lon}) -> .a;
        (
          area.a[admin_level="8"]; area.a[admin_level="6"]; area.a[admin_level="4"]; area.a[admin_level="2"];
        );
        out tags;
        """
        
        try:
            response = requests.post(self.overpass_url, data=query, timeout=DEFAULT_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            found_levels = {}
            for element in data.get('elements', []): # Corrected indentation
                tags = element.get('tags', {})
                level_str = tags.get('admin_level')
                name = tags.get('name') or tags.get('name:en')
                if level_str and name and level_str.isdigit():
                    level = int(level_str)
                    level_key = f'level_{level}_name'
                    if level_key in hierarchy:
                        hierarchy[level_key] = name
                        found_levels[level] = name # Store level number and name
            
            # Determine containing and parent levels
            sorted_found_levels = sorted(found_levels.keys(), reverse=True) # e.g., [8, 6, 4, 2]
            
            if sorted_found_levels:
                hierarchy['containing_level'] = sorted_found_levels[0] # Most specific level found
                if len(sorted_found_levels) > 1:
                    hierarchy['parent_level'] = sorted_found_levels[1] # Next level up
                    hierarchy['parent_name'] = found_levels.get(sorted_found_levels[1])
            
            return hierarchy
            
        except requests.exceptions.RequestException as e:
            # ... (Error handling remains the same) ...
            return hierarchy
        except Exception as e:
            # ... (Error handling remains the same) ...
            return hierarchy
    # --- End NEW Function ---

    # --- Re-added save_osm_data_to_excel_only Function --- 
    def save_osm_data_to_excel_only(self, locations: List[Dict], filename: str):
        """Saves initially collected OSM data and hierarchy to an Excel file."""
        if not locations:
            print("No OSM locations found to save.")
            return False
        
        try:
            # Normalize the data, including the admin_hierarchy nested dict
            df = pd.json_normalize(locations, sep='_')
            
            # Define expected columns (OSM tags + Hierarchy)
            core_columns = ['name', 'type', 'latitude', 'longitude']
            osm_tags_columns = ['osm_population_tag', 'osm_population_date', 'wikidata_id', 'wikipedia_link']
            # Add hierarchy columns (handle potential missing keys from normalize)
            hierarchy_columns = ['admin_hierarchy_containing_level', 'admin_hierarchy_parent_level', 'admin_hierarchy_parent_name',
                                 'admin_hierarchy_level_8_name', 'admin_hierarchy_level_6_name', 
                                 'admin_hierarchy_level_4_name', 'admin_hierarchy_level_2_name']

            column_order = core_columns + osm_tags_columns + hierarchy_columns
            
            # Ensure all expected columns exist
            for col in column_order:
                if col not in df.columns:
                    df[col] = None
            
            # Select and reorder final columns
            df = df[[col for col in column_order if col in df.columns]]

            # Clean up hierarchy column names
            df.columns = [col.replace('admin_hierarchy_','') for col in df.columns]

            # Convert relevant columns to numeric
            numeric_cols = ['osm_population_tag', 'containing_level', 'parent_level']
            for num_col in numeric_cols:
                 if num_col in df.columns:
                      df[num_col] = pd.to_numeric(df[num_col], errors='coerce').astype('Float64')
            
            # Save to Excel
            df.to_excel(filename, index=False, engine='openpyxl')
            return True
        except Exception as e:
            print(f"Error saving OSM-only data to Excel ({filename}): {str(e)}")
            traceback.print_exc()
            return False
    # --- End Function Update --- 

def create_config_file():
    """
    Creates a configuration file with default settings.
    
    Returns:
        bool: True if configuration file was created successfully, False otherwise
    """
    config = configparser.ConfigParser()
    
    # File paths section
    config['Paths'] = {
        'default_kmz_file': DEFAULT_KMZ_FILE,
        'default_excel_file': DEFAULT_EXCEL_FILE,
        'default_verification_map': DEFAULT_VERIFICATION_MAP,
        'help': 'Default file paths used by the script'
    }
    
    # Place types section
    config['PlaceTypes'] = {
        'primary_types': 'city, town, district, county, municipality, borough, suburb',
        'additional_types': 'neighbourhood, village, locality',
        'special_types': '',  # Removed: commercial_area
        'help': 'Types of places to search for in OpenStreetMap'
    }
    
    # OpenAI API Configuration
    api_key_from_env = os.environ.get(OPENAI_API_ENV_VAR, '')
    config['OpenAI'] = {
        'api_key': api_key_from_env if api_key_from_env else 'YOUR_API_KEY_HERE',
        'use_env_var': 'true',  # Whether to use environment variable instead of config file
        'chunk_size': str(DEFAULT_CHUNK_SIZE),
        'timeout': str(DEFAULT_API_TIMEOUT),
        'max_retries': str(DEFAULT_MAX_RETRIES),
        'max_locations': str(DEFAULT_MAX_LOCATIONS),
        'pause_before_gpt': str(DEFAULT_PAUSE_BEFORE_GPT).lower(),
        'enable_web_browsing': str(DEFAULT_ENABLE_WEB_BROWSING).lower(),
        'help': '''Settings for OpenAI GPT population estimation.
For security, it is recommended to use the OPENAI_API_ENV_VAR environment variable rather than storing the key in this file.
Web browsing capabilities require GPT-4 API access with appropriate permissions.'''
    }
    
    # Write to file
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        
        print(f"Configuration file created successfully at: {config_path}")
        print("You can now edit this file to customize the script behavior.")
        if api_key_from_env:
            print(f"Your OpenAI API key from environment variable {OPENAI_API_ENV_VAR} has been set in the config.")
        else:
            print(f"For security, you can set your OpenAI API key as environment variable {OPENAI_API_ENV_VAR}")
            print("Or edit the config file directly to set the 'api_key' value in the [OpenAI] section.")
        print("\nWeb Browsing Requirements:")
        print("1. GPT-4 API access with web browsing capabilities")
        print("2. Appropriate API key permissions")
        print("3. Sufficient API credits/usage limits")
        print("4. Enable web browsing in the config file or via command line")
        
        return True
    except Exception as e:
        print(f"Error creating configuration file: {str(e)}")
        return False

def read_config():
    """
    Read configuration from config file.
    
    Returns:
        dict: Configuration values
    """
    config_values = {}
    
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Get OpenAI settings
            if 'OpenAI' in config:
                # Check if we should use environment variable
                use_env_var = config['OpenAI'].getboolean('use_env_var', True)
                
                # Try to get API key from environment first if enabled
                api_key = None
                if use_env_var:
                    api_key = os.environ.get(OPENAI_API_ENV_VAR)
                    if api_key:
                        print(f"Using OpenAI API key from environment variable {OPENAI_API_ENV_VAR}")
                
                # Fall back to config file if not in environment
                if not api_key:
                    api_key = config['OpenAI'].get('api_key', '')
                    if api_key and api_key != 'YOUR_API_KEY_HERE':
                        print("Using OpenAI API key from config file")
                    else:
                        api_key = ''
                
                config_values['openai_api_key'] = api_key
                config_values['chunk_size'] = config['OpenAI'].getint('chunk_size', DEFAULT_CHUNK_SIZE)
                config_values['max_locations'] = config['OpenAI'].getint('max_locations', DEFAULT_MAX_LOCATIONS)
                config_values['pause_before_gpt'] = config['OpenAI'].getboolean('pause_before_gpt', DEFAULT_PAUSE_BEFORE_GPT)
                config_values['enable_web_browsing'] = config['OpenAI'].getboolean('enable_web_browsing', DEFAULT_ENABLE_WEB_BROWSING)
            
            print(f"Loaded configuration from {config_path}")
        else:
            print(f"Configuration file not found: {config_path}")
            
            # Try environment variable as fallback even without config file
            api_key = os.environ.get(OPENAI_API_ENV_VAR, '')
            if api_key:
                print(f"Using OpenAI API key from environment variable {OPENAI_API_ENV_VAR}")
                config_values['openai_api_key'] = api_key
            
    except Exception as e:
        print(f"Error reading configuration: {str(e)}")
    
    return config_values

def main():
    parser = argparse.ArgumentParser(description='Find locations within a KMZ boundary using OpenStreetMap.')
    parser.add_argument('kmz_file', nargs='?', default=DEFAULT_KMZ_FILE,
                        help=f'Path to the KMZ file (default: {DEFAULT_KMZ_FILE})')
    parser.add_argument('-o', '--output', default=DEFAULT_EXCEL_FILE,
                        help=f'Path to save the output Excel file (default: {DEFAULT_EXCEL_FILE})')
    parser.add_argument('-m', '--map', default=DEFAULT_VERIFICATION_MAP,
                        help=f'Path to save the verification map (currently ignored)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-c', '--create-config', action='store_true',
                        help='Create a new config file with default settings and exit')
    parser.add_argument('-g', '--use-gpt', action='store_true',
                        help='Enable population estimation using OpenAI GPT')
    parser.add_argument('-k', '--api-key', 
                        help='OpenAI API key (overrides config/env variable)')
    parser.add_argument('--chunk-size', type=int, # Default comes from constant now
                        help=f'Number of locations per GPT batch (default: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--max-locations', type=int, # Default comes from constant now
                        help=f'Maximum number of locations to process (default: {DEFAULT_MAX_LOCATIONS}, 0 means no limit)')
    parser.add_argument('--pause-before-gpt', action='store_true', default=None, # Default None to check config
                        help='Pause script before the first GPT batch for user review')
    parser.add_argument('--enable-web-browsing', action=argparse.BooleanOptionalAction, default=None, # Allows --enable-web-browsing / --no-enable-web-browsing
                         help='Enable/disable web browsing for GPT (requires capable model & API access)')
    parser.add_argument('--export-osm-only', action='store_true', # <-- Re-added argument
                        help='Export raw collected OSM data (selected tags) and exit immediately.')

    args = parser.parse_args()
    
    # Create config file if requested
    if args.create_config:
        create_config_file() # Assume this function exists and works
        sys.exit(0) # Exit after creating config
    
    # Read configuration from file first
    config = read_config() # Assume this function exists and works
    
    # --- Determine effective settings (Args > Config > Defaults) ---
    
    # API Key: Arg -> Env Var -> Config -> Default ('')
    openai_api_key = args.api_key if args.api_key is not None else os.environ.get(OPENAI_API_ENV_VAR, config.get('openai_api_key', ''))

    # Use GPT: Arg (--use-gpt) takes precedence
    use_gpt = args.use_gpt # If arg is present, it's True, otherwise False (standard argparse)
    
    # Chunk Size: Arg -> Config -> Default
    chunk_size = args.chunk_size if args.chunk_size is not None else config.get('chunk_size', DEFAULT_CHUNK_SIZE)
    
    # Max Locations: Arg -> Config -> Default
    max_locations = args.max_locations if args.max_locations is not None else config.get('max_locations', DEFAULT_MAX_LOCATIONS)

    # Pause before GPT: Arg -> Config -> Default
    # Need careful handling since action='store_true' defaults to False if not present
    pause_before_gpt = args.pause_before_gpt if args.pause_before_gpt is not None else config.get('pause_before_gpt', DEFAULT_PAUSE_BEFORE_GPT)

    # Enable Web Browsing: Arg -> Config -> Default
    enable_web_browsing = args.enable_web_browsing if args.enable_web_browsing is not None else config.get('enable_web_browsing', DEFAULT_ENABLE_WEB_BROWSING)


    # Final check: If using GPT, ensure API key exists
    if use_gpt and not openai_api_key:
        print("\nWarning: GPT population estimation is enabled (--use-gpt), but no OpenAI API key was found.")
        print(f"  Please provide one via --api-key argument, the {OPENAI_API_ENV_VAR} environment variable, or in the config.ini file.")
        print("  Disabling GPT estimation for this run.")
        use_gpt = False # Disable GPT if no key

    # --- Instantiate and Run Analyzer ---
    try:
        analyzer = LocationAnalyzer(
            kmz_file=args.kmz_file, 
            output_excel=args.output,
            map_output=args.map,
            verbose=args.verbose,
            openai_api_key=openai_api_key,
            use_gpt=use_gpt,
            chunk_size=chunk_size,
            max_locations=max_locations,
            pause_before_gpt=pause_before_gpt,
            enable_web_browsing=enable_web_browsing,
            export_osm_only=args.export_osm_only # <-- Re-added passing the flag
        )
        success = analyzer.run()
        sys.exit(0 if success else 1)
        
    except FileNotFoundError as e:
         print(f"Error: Input file not found - {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        traceback.print_exc() # Print detailed traceback for unexpected errors
        sys.exit(1)

if __name__ == "__main__":
    main()
