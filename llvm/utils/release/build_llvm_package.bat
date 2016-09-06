@echo off
setlocal

REM Script for building the LLVM installer on Windows,
REM used for the the weekly snapshots at http://www.llvm.org/builds.
REM
REM Usage: build_llvm_package.bat <revision>

REM Prerequisites:
REM
REM   Visual Studio 2013, CMake, Ninja, SVN, GNUWin32,
REM   NSIS with the strlen_8192 patch,
REM   Visual Studio 2013 SDK (for the clang-format plugin).


REM You may need to modify the paths below:
set vcdir=c:\Program Files (x86)\Microsoft Visual Studio 12.0\VC
set PATH=%PATH%;c:\gnuwin32\bin

set revision=%1
set branch=trunk
set package_version=4.0.0-r%revision%
set clang_format_vs_version=4.0.0.%revision%
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


REM Setting CMAKE_CL_SHOWINCLUDES_PREFIX to work around PR27226.
set cmake_flags=-DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_TOOLCHAIN_ONLY=ON -DLLVM_USE_CRT_RELEASE=MT -DCLANG_FORMAT_VS_VERSION=%clang_format_vs_version% -DPACKAGE_VERSION=%package_version% -DCMAKE_CL_SHOWINCLUDES_PREFIX="Note: including file: "

REM TODO: Run all tests, including lld and compiler-rt.

call "%vcdir%/vcvarsall.bat" x86
set CC=
set CXX=
mkdir build32_stage0
cd build32_stage0
cmake -GNinja %cmake_flags% ..\llvm || exit /b
ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang ||  exit /b
cd..

mkdir build32
cd build32
set CC=..\build32_stage0\bin\clang-cl
set CXX=..\build32_stage0\bin\clang-cl
cmake -GNinja %cmake_flags% -DBUILD_CLANG_FORMAT_VS_PLUGIN=ON ..\llvm || exit /b
ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang ||  exit /b
copy ..\llvm\tools\clang\tools\clang-format-vs\ClangFormat\bin\Release\ClangFormat.vsix ClangFormat-r%revision%.vsix
ninja package || exit /b
cd ..


call "%vcdir%/vcvarsall.bat" amd64
set CC=
set CXX=
mkdir build64_stage0
cd build64_stage0
cmake -GNinja %cmake_flags%  ..\llvm || exit /b
ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang ||  exit /b
cd..

mkdir build64
cd build64
set CC=..\build64_stage0\bin\clang-cl
set CXX=..\build64_stage0\bin\clang-cl
cmake -GNinja %cmake_flags% ..\llvm || exit /b
ninja all || exit /b
ninja check || ninja check || ninja check || exit /b
ninja check-clang || ninja check-clang || ninja check-clang ||  exit /b
ninja package || exit /b
cd ..
