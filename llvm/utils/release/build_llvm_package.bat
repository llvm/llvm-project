@echo off
setlocal

REM Script for building the LLVM installer on Windows,
REM used for the the weekly snapshots at http://www.llvm.org/builds.
REM
REM Usage: build_llvm_package.bat <revision>

REM Prerequisites:
REM
REM   Visual Studio 2017, CMake, Ninja, SVN, GNUWin32, SWIG, Python 3,
REM   NSIS with the strlen_8192 patch,
REM   Visual Studio 2017 SDK and Nuget (for the clang-format plugin),
REM   Perl (for the OpenMP run-time).
REM
REM
REM   For LLDB, SWIG version <= 3.0.8 needs to be used to work around
REM   https://github.com/swig/swig/issues/769


REM You need to modify the paths below:
set vsdevcmd=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\Tools\VsDevCmd.bat

set python32_dir=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python36-32
set python64_dir=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python36

set revision=%1
set branch=trunk
set package_version=9.0.0-r%revision%
set clang_format_vs_version=9.0.0.%revision%
set build_dir=llvm_package_%revision%

echo Branch: %branch%
echo Revision: %revision%
echo Package version: %package_version%
echo Clang format plugin version: %clang_format_vs_version%
echo Build dir: %build_dir%
echo.
pause

mkdir %build_dir%
cd %build_dir%

echo Checking out %branch% at r%revision%...
svn.exe export -r %revision% http://llvm.org/svn/llvm-project/llvm/%branch% llvm || exit /b
svn.exe export -r %revision% http://llvm.org/svn/llvm-project/cfe/%branch% llvm/tools/clang || exit /b
svn.exe export -r %revision% http://llvm.org/svn/llvm-project/clang-tools-extra/%branch% llvm/tools/clang/tools/extra || exit /b
svn.exe export -r %revision% http://llvm.org/svn/llvm-project/lld/%branch% llvm/tools/lld || exit /b
svn.exe export -r %revision% http://llvm.org/svn/llvm-project/compiler-rt/%branch% llvm/projects/compiler-rt || exit /b
svn.exe export -r %revision% http://llvm.org/svn/llvm-project/openmp/%branch% llvm/projects/openmp || exit /b
svn.exe export -r %revision% http://llvm.org/svn/llvm-project/lldb/%branch% llvm/tools/lldb || exit /b


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
	-DCMAKE_CL_SHOWINCLUDES_PREFIX="Note: including file: "

REM TODO: Run the "check-all" tests.

set "VSCMD_START_DIR=%CD%"
call "%vsdevcmd%" -arch=x86
set CC=
set CXX=
mkdir build32_stage0
cd build32_stage0
REM Work around VS2017 bug by using MinSizeRel.
cmake -GNinja %cmake_flags% -DPYTHON_HOME=%python32_dir% -DCMAKE_BUILD_TYPE=MinSizeRel ..\llvm || exit /b
ninja all || ninja all || ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang || exit /b
ninja check-lld || ninja check-lld || ninja check-lld || exit /b
ninja check-sanitizer || ninja check-sanitizer || ninja check-sanitizer || exit /b
cd..

mkdir build32
cd build32
set CC=..\build32_stage0\bin\clang-cl
set CXX=..\build32_stage0\bin\clang-cl
cmake -GNinja %cmake_flags% -DPYTHON_HOME=%python32_dir% ..\llvm || exit /b
ninja all || ninja all || ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang || exit /b
ninja check-lld || ninja check-lld || ninja check-lld || exit /b
ninja check-sanitizer || ninja check-sanitizer || ninja check-sanitizer || exit /b
ninja package || exit /b
cd ..

REM The plug-in is built separately as it uses a statically linked clang-format.exe.
mkdir build_vsix
cd build_vsix
set CC=..\build32_stage0\bin\clang-cl
set CXX=..\build32_stage0\bin\clang-cl
cmake -GNinja %cmake_flags% -DLLVM_USE_CRT_RELEASE=MT -DBUILD_CLANG_FORMAT_VS_PLUGIN=ON -DPYTHON_HOME=%python32_dir% ..\llvm || exit /b
ninja clang_format_vsix || exit /b
copy ..\llvm\tools\clang\tools\clang-format-vs\ClangFormat\bin\Release\ClangFormat.vsix ClangFormat-r%revision%.vsix
cd ..


set "VSCMD_START_DIR=%CD%"
call "%vsdevcmd%" -arch=amd64
set CC=
set CXX=
mkdir build64_stage0
cd build64_stage0
REM Work around VS2017 bug by using MinSizeRel.
cmake -GNinja %cmake_flags% -DPYTHON_HOME=%python64_dir% -DCMAKE_BUILD_TYPE=MinSizeRel ..\llvm || exit /b
ninja all || ninja all || ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang || exit /b
ninja check-lld || ninja check-lld || ninja check-lld || exit /b
ninja check-sanitizer || ninja check-sanitizer || ninja check-sanitizer || exit /b
cd..

mkdir build64
cd build64
set CC=..\build64_stage0\bin\clang-cl
set CXX=..\build64_stage0\bin\clang-cl
cmake -GNinja %cmake_flags% -DPYTHON_HOME=%python64_dir% ..\llvm || exit /b
ninja all || ninja all || ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang || exit /b
ninja check-lld || ninja check-lld || ninja check-lld || exit /b
ninja check-sanitizer || ninja check-sanitizer || ninja check-sanitizer || exit /b
ninja package || exit /b
cd ..
