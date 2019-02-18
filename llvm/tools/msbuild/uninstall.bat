@echo off

echo Uninstalling MSVC integration...

REM In general this script should not be used except for development and testing
REM purposes.  The proper way to install is via the VSIX, and the proper way to
REM uninstall is through the Visual Studio extension manager.

REM CD to the directory of this batch file.
cd /d %~dp0

SET VCTargets=%ProgramFiles(x86)%\Microsoft Visual Studio\2017\Professional\Common7\IDE\VC\VCTargets

ECHO Uninstalling Common Files
IF EXIST "%VCTargets%\LLVM.Cpp.Common.props" del "%VCTargets%\LLVM.Cpp.Common.props"
IF EXIST "%VCTargets%\LLVM.Cpp.Common.targets" del "%VCTargets%\LLVM.Cpp.Common.targets"

ECHO Uninstalling x64 Platform Toolset
SET PlatformToolsets=%VCTargets%\Platforms\x64\PlatformToolsets
IF EXIST "%PlatformToolsets%\llvm\Toolset.props" del "%PlatformToolsets%\llvm\Toolset.props"
IF EXIST "%PlatformToolsets%\llvm\Toolset.targets" del "%PlatformToolsets%\llvm\Toolset.targets"
IF EXIST "%PlatformToolsets%\llvm" rd "%PlatformToolsets%\llvm"

ECHO Uninstalling Win32 Platform Toolset
SET PlatformToolsets=%VCTargets%\Platforms\Win32\PlatformToolsets
IF EXIST "%PlatformToolsets%\llvm\Toolset.props" del "%PlatformToolsets%\llvm\Toolset.props"
IF EXIST "%PlatformToolsets%\llvm\Toolset.targets" del "%PlatformToolsets%\llvm\Toolset.targets"
IF EXIST "%PlatformToolsets%\llvm" rd "%PlatformToolsets%\llvm"

ECHO Uninstalling C++ Settings UI
IF EXIST "%VCTargets%\1033\llvm-general.xml" del "%VCTargets%\1033\llvm-general.xml"

echo Done!
