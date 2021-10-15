@echo off
setlocal

REM Script for building the LLVM installer on Windows,
REM used for the the weekly snapshots at http://www.llvm.org/builds.
REM
REM Usage: build_llvm_package.bat <revision>

REM Prerequisites:
REM
REM   Visual Studio 2019, CMake, Ninja, GNUWin32, SWIG, Python 3,
REM   NSIS with the strlen_8192 patch,
REM   Visual Studio 2019 SDK and Nuget (for the clang-format plugin),
REM   Perl (for the OpenMP run-time), 7Zip.
REM
REM
REM   For LLDB, SWIG version <= 3.0.8 needs to be used to work around
REM   https://github.com/swig/swig/issues/769


REM You need to modify the paths below:
set vsdevcmd=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\VsDevCmd.bat

set python32_dir=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python36-32
set python64_dir=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python36

for /f "usebackq" %%i in (`PowerShell ^(Get-Date^).ToString^('yyyyMMdd'^)`) do set datestamp=%%i

set revision=%1
set package_version=13.0.1-%revision:~0,8%
set clang_format_vs_version=13.0.1.%datestamp%
set build_dir=llvm_package_%revision:~0,8%

echo Revision: %revision%
echo Package version: %package_version%
echo Clang format plugin version: %clang_format_vs_version%
echo Build dir: %build_dir%
echo.
pause

mkdir %build_dir%
cd %build_dir%

echo Checking out %revision%
curl -L https://github.com/llvm/llvm-project/archive/%revision%.zip -o src.zip || exit /b
7z x src.zip || exit /b
mv llvm-project-* llvm-project || exit /b

REM Setting CMAKE_CL_SHOWINCLUDES_PREFIX to work around PR27226.
set cmake_flags=^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_ENABLE_ASSERTIONS=ON ^
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=ON ^
  -DLLVM_BUILD_LLVM_C_DYLIB=ON ^
  -DCMAKE_INSTALL_UCRT_LIBRARIES=ON ^
  -DCLANG_FORMAT_VS_VERSION=%clang_format_vs_version% ^
  -DPACKAGE_VERSION=%package_version% ^
  -DLLDB_RELOCATABLE_PYTHON=1 ^
  -DLLDB_TEST_COMPILER=%cd%\build32_stage0\bin\clang.exe ^
  -DCMAKE_CL_SHOWINCLUDES_PREFIX="Note: including file: " ^
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;compiler-rt;openmp;lldb"

REM TODO: Run the "check-all" tests.

set "VSCMD_START_DIR=%CD%"
call "%vsdevcmd%" -arch=x86
set CC=
set CXX=
mkdir build32_stage0
cd build32_stage0
cmake -GNinja %cmake_flags% -DPYTHON_HOME=%python32_dir% -DPYTHON_EXECUTABLE=%python32_dir%\python.exe ..\llvm-project\llvm || exit /b
ninja all || ninja all || ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang || exit /b
ninja check-lld || ninja check-lld || ninja check-lld || exit /b
ninja check-sanitizer || ninja check-sanitizer || ninja check-sanitizer || exit /b
ninja check-clang-tools || ninja check-clang-tools || ninja check-clang-tools || exit /b
ninja check-clangd || ninja check-clangd || ninja check-clangd || exit /b
cd..

mkdir build32
cd build32
set CC=..\build32_stage0\bin\clang-cl
set CXX=..\build32_stage0\bin\clang-cl
cmake -GNinja %cmake_flags% -DPYTHON_HOME=%python32_dir% -DPYTHON_EXECUTABLE=%python32_dir%\python.exe ..\llvm-project\llvm || exit /b
ninja all || ninja all || ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang || exit /b
ninja check-lld || ninja check-lld || ninja check-lld || exit /b
ninja check-sanitizer || ninja check-sanitizer || ninja check-sanitizer || exit /b
ninja check-clang-tools || ninja check-clang-tools || ninja check-clang-tools || exit /b
ninja check-clangd || ninja check-clangd || ninja check-clangd || exit /b
ninja package || exit /b

7z x LLVM-%package_version%-win32.exe -orepack
rmdir /s /q repack\$PLUGINSDIR
del repack\Uninstall.exe
7z a LLVM-%package_version%-win32.zip .\repack\* -mx9
cd ..


REM The plug-in is built separately as it uses a statically linked clang-format.exe.
mkdir build_vsix
cd build_vsix
REM Having VSSDKINSTALL set makes devenv *not* find the SDK for some reason.
set VSSDKINSTALL=
set CC=..\build32_stage0\bin\clang-cl
set CXX=..\build32_stage0\bin\clang-cl
cmake -GNinja %cmake_flags% -DLLVM_USE_CRT_RELEASE=MT -DBUILD_CLANG_FORMAT_VS_PLUGIN=ON -DPYTHON_HOME=%python32_dir% -DPYTHON_EXECUTABLE=%python32_dir%\python.exe ..\llvm-project\llvm || exit /b
ninja clang_format_vsix || exit /b
copy ..\llvm-project\llvm\tools\clang\tools\clang-format-vs\ClangFormat\bin\Release\ClangFormat.vsix ClangFormat-r%revision%.vsix
cd ..


set "VSCMD_START_DIR=%CD%"
call "%vsdevcmd%" -arch=amd64
set CC=
set CXX=
mkdir build64_stage0
cd build64_stage0
cmake -GNinja %cmake_flags% -DPYTHON_HOME=%python64_dir% -DPYTHON_EXECUTABLE=%python64_dir%\python.exe ..\llvm-project\llvm || exit /b
ninja all || ninja all || ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang || exit /b
ninja check-lld || ninja check-lld || ninja check-lld || exit /b
ninja check-sanitizer || ninja check-sanitizer || ninja check-sanitizer || exit /b
ninja check-clang-tools || ninja check-clang-tools || ninja check-clang-tools || exit /b
ninja check-clangd || ninja check-clangd || ninja check-clangd || exit /b
cd..

mkdir build64
cd build64
set CC=..\build64_stage0\bin\clang-cl
set CXX=..\build64_stage0\bin\clang-cl
cmake -GNinja %cmake_flags% -DPYTHON_HOME=%python64_dir% -DPYTHON_EXECUTABLE=%python64_dir%\python.exe ..\llvm-project\llvm || exit /b
ninja all || ninja all || ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang || exit /b
ninja check-lld || ninja check-lld || ninja check-lld || exit /b
ninja check-sanitizer || ninja check-sanitizer || ninja check-sanitizer || exit /b
ninja check-clang-tools || ninja check-clang-tools || ninja check-clang-tools || exit /b
ninja check-clangd || ninja check-clangd || ninja check-clangd || exit /b
ninja package || exit /b

7z x LLVM-%package_version%-win64.exe -orepack
rmdir /s /q repack\$PLUGINSDIR
del repack\Uninstall.exe
7z a LLVM-%package_version%-win64.zip .\repack\* -mx9
cd ..
