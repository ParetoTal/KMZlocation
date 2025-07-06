@echo off
echo 🚀 Starting Location Analyzer Installation...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python from python.org first.
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install Python from python.org first.
    pause
    exit /b 1
)

:: Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Install requirements
echo 📥 Installing requirements...
pip install -r requirements.txt

:: Create desktop shortcut
echo 🔗 Creating desktop shortcut...
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\Location Analyzer.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%~dp0run_analyzer.bat" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%~dp0" >> CreateShortcut.vbs
echo oLink.Description = "Location Analyzer" >> CreateShortcut.vbs
echo oLink.IconLocation = "%~dp0app.ico" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript //nologo CreateShortcut.vbs
del CreateShortcut.vbs

:: Create run script
echo @echo off > run_analyzer.bat
echo call venv\Scripts\activate.bat >> run_analyzer.bat
echo streamlit run app.py >> run_analyzer.bat

echo ✅ Installation complete!
echo 📌 You can now run the Location Analyzer by double-clicking the 'Location Analyzer' icon on your desktop.
echo 🌐 The application will open in your web browser automatically.
echo.
echo Proudly built by ParetoLeads.com
pause 