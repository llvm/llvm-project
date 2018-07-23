@echo off

echo Installing MSVC integration...
set SUCCESS=0

REM In general this script should not be used except for development and testing
REM purposes.  The proper way to install is via the VSIX, and the proper way to
REM uninstall is through the Visual Studio extension manager.

REM Change to the directory of this batch file.
cd /d %~dp0

REM Older versions of VS would look for these files in the Program Files\MSBuild directory
REM but with VS2017 it seems to look for these directly in the Visual Studio instance.
REM This means we'll need to do a little extra work to properly detect all the various
REM instances, but in reality we can probably sidestep all of this by just wrapping this
REM in a vsix and calling it a day, as that should handle everything for us.
SET VCTargets=%ProgramFiles(x86)%\Microsoft Visual Studio\2017\Professional\Common7\IDE\VC\VCTargets

ECHO Installing Common Files
copy LLVM.Cpp.Common.props "%VCTargets%"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy LLVM.Cpp.Common.targets "%VCTargets%"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED

ECHO Installing x64 Platform Toolset
SET PlatformToolsets=%VCTargets%\Platforms\x64\PlatformToolsets
IF NOT EXIST "%PlatformToolsets%\llvm" mkdir "%PlatformToolsets%\llvm"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy PlatformX64\Toolset.props "%PlatformToolsets%\llvm"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy PlatformX64\Toolset.targets "%PlatformToolsets%\llvm"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED

ECHO Installing Win32 Platform Toolset
SET PlatformToolsets=%VCTargets%\Platforms\Win32\PlatformToolsets
IF NOT EXIST "%PlatformToolsets%\llvm" mkdir "%PlatformToolsets%\llvm"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy PlatformX86\Toolset.props "%PlatformToolsets%\llvm"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy PlatformX86\Toolset.targets "%PlatformToolsets%\llvm"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED

ECHO Installing C++ Settings UI
copy llvm-general.xml "%VCTargets%\1033"
IF NOT %ERRORLEVEL% == 0 GOTO FAILED

:DONE
echo Done!
goto END

:FAILED
echo MSVC integration install failed.
pause
goto END

:END
