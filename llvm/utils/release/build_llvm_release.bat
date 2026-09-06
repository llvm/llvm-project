@echo off

REM Filter out tests that are known to fail.
set "LIT_FILTER_OUT=gh110231.cpp|crt_initializers.cpp|init-order-atexit.cpp|use_after_return_linkage.cpp|initialization-bug.cpp|initialization-bug-no-global.cpp|trace-malloc-unbalanced.test|trace-malloc-2.test|TraceMallocTest|TestLockFileExclusive"

setlocal enabledelayedexpansion
goto begin

:usage
echo Script for building the LLVM installer on Windows,
echo used for the releases at https://github.com/llvm/llvm-project/releases
echo.
echo Usage: build_llvm_release.bat --version ^<version^> [--x86,--x64, --arm64] [--skip-checkout] [--local-python] [--force-msvc] [--fast-build] [--enhanced-pgo] [--enable-thinlto] [--enable-pdb]
echo.
echo Options:
echo --version: [required] version to build
echo --help: display this help
echo --x86: build and test x86 variant
echo --x64: build and test x64 variant
echo --arm64: build and test arm64 variant
echo --skip-checkout: use local git checkout instead of downloading src.zip
echo --local-python: use installed Python and does not try to use a specific version (3.11)
echo --force-msvc: use MSVC compiler for stage0, even if clang-cl is present
echo --enhanced-pgo: train the instrumented stage1 clang by building LLVMSupport instead of
echo   the legacy single-file Sema.cpp training step, and use the resulting profile for
echo   stage2 (64-bit builds only).
echo --enable-thinlto: build stage2 with ThinLTO (64-bit builds only)
echo --enable-pdb: generate PDB debug info files for stage2 and include them as an additional artifact (64-bit builds only)
echo.
echo Note: At least one variant to build is required.
echo.
echo Example: build_llvm_release.bat --version 15.0.0 --x86 --x64
exit /b 1


:begin

::==============================================================================
:: parse args
set version=
set help=
set x86=
set x64=
set arm64=
set skip-checkout=
set local-python=
set force-msvc=
set fast-build=
set enhanced-pgo=
set enable-thinlto=
set enable-pdb=
call :parse_args %*

if "%help%" NEQ "" goto usage

if "%version%" == "" (
    echo --version option is required
    echo =============================
    goto usage
)

if "%arm64%" == "" if "%x64%" == "" if "%x86%" == "" (
    echo nothing to build!
    echo choose one or several variants from: --x86 --x64 --arm64
    exit /b 1
)

::==============================================================================
:: check prerequisites
REM Note:
REM   7zip versions 21.x and higher will try to extract the symlinks in
REM   llvm's git archive, which requires running as administrator.

REM Check 7-zip version and/or administrator permissions.
for /f "delims=" %%i in ('7z.exe ^| findstr /r "2[1-9].[0-9][0-9]"') do set version_7z=%%i
if not "%version_7z%"=="" (
  REM Unique temporary filename to use by the 'mklink' command.
  set "link_name=%temp%\%username%_%random%_%random%.tmp"

  REM As the 'mklink' requires elevated permissions, the symbolic link
  REM creation will fail if the script is not running as administrator.
  mklink /d "!link_name!" . 1>nul 2>nul
  if errorlevel 1 (
    echo.
    echo Script requires administrator permissions, or a 7-zip version 20.x or older.
    echo Current version is "%version_7z%"
    exit /b 1
  ) else (
    REM Remove the temporary symbolic link.
    rd "!link_name!"
  )
)

REM Prerequisites:
REM
REM   Visual Studio 2019, CMake, Ninja, GNUWin32, SWIG, Python 3,
REM   Perl (for the OpenMP run-time).
REM
REM
REM   For LLDB, SWIG version 4.1.1 should be used.
REM

:: Detect Visual Studio
set vsinstall=
set vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe

if "%VSINSTALLDIR%" NEQ "" (
  echo using enabled Visual Studio installation
  set "vsinstall=%VSINSTALLDIR%"
) else (
  echo using vswhere to detect Visual Studio installation
  FOR /F "delims=" %%r IN ('^""%vswhere%" -nologo -latest -products "*" -all -property installationPath^"') DO set vsinstall=%%r
)
set "vsdevcmd=%vsinstall%\Common7\Tools\VsDevCmd.bat"

if not exist "%vsdevcmd%" (
  echo Can't find any installation of Visual Studio
  exit /b 1
)
echo Using VS devcmd: %vsdevcmd%

::==============================================================================
:: start echoing what we do
@echo on

set python32_dir=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311-32
set python64_dir=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311
set pythonarm64_dir=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311-arm64

set revision=llvmorg-%version%
set package_version=%version%
set build_dir=%cd%\llvm_package_%package_version%

echo Revision: %revision%
echo Package version: %package_version%
echo Build dir: %build_dir%
echo.

if exist %build_dir% (
  echo Build directory already exists: %build_dir%
  exit /b 1
)
mkdir %build_dir%
cd %build_dir% || exit /b 1

if "%skip-checkout%" == "true" (
  echo Using local source
  set llvm_src=%~dp0..\..\..
) else (
  echo Checking out %revision%
  curl -L https://github.com/llvm/llvm-project/archive/%revision%.zip -o src.zip || exit /b 1
  7z x src.zip || exit /b 1
  mv llvm-project-* llvm-project || exit /b 1
  set llvm_src=%build_dir%\llvm-project
)

set libxml_version=2.9.12
curl -O https://gitlab.gnome.org/GNOME/libxml2/-/archive/v%libxml_version%/libxml2-v%libxml_version%.tar.gz || exit /b 1
call :verify_checksum libxml2-v%libxml_version%.tar.gz sha256 98bfa7a9a5e2a75638422050740448ee9f02bf4dc2075c9822d7747d5ff9e617 || exit /b 1
REM 'test' directory excluded because of symlinks.
tar zxf libxml2-v%libxml_version%.tar.gz --exclude "test/*" || exit /b 1

REM FIXME: It would be preferrable to use zlib-ng here since it is better
REM        maintained and performs better than zlib, but lld tests currently
REM        assume the original zlib is used. They need to be fixed first:
REM        https://github.com/llvm/llvm-project/pull/186630#discussion_r2939953952
set zlib_version=1.3.2
curl -LO https://github.com/madler/zlib/releases/download/v%zlib_version%/zlib-%zlib_version%.tar.gz || exit /b 1
call :verify_checksum zlib-%zlib_version%.tar.gz sha256 bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16 || exit /b 1
tar zxf zlib-%zlib_version%.tar.gz || exit /b 1

set zstd_version=1.5.7
curl -LO https://github.com/facebook/zstd/releases/download/v%zstd_version%/zstd-%zstd_version%.tar.gz || exit /b 1
call :verify_checksum zstd-%zstd_version%.tar.gz sha256 eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3 || exit /b 1
REM 'tests' directory excluded because of symlinks.
tar zxf zstd-%zstd_version%.tar.gz --exclude "tests/*" || exit /b 1

REM Setting CMAKE_CL_SHOWINCLUDES_PREFIX to work around PR27226.
REM Common flags for all builds.
set common_compiler_flags=-DLIBXML_STATIC
set common_cmake_flags=^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_ENABLE_ASSERTIONS=OFF ^
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=ON ^
  -DLLVM_TARGETS_TO_BUILD="AArch64;ARM;X86;BPF;WebAssembly;RISCV;NVPTX" ^
  -DLLVM_BUILD_LLVM_C_DYLIB=ON ^
  -DPython3_FIND_REGISTRY=NEVER ^
  -DPACKAGE_VERSION=%package_version% ^
  -DCMAKE_CL_SHOWINCLUDES_PREFIX="Note: including file: " ^
  -DLLVM_ENABLE_LIBXML2=FORCE_ON ^
  -DCLANG_ENABLE_LIBXML2=OFF ^
  -DLLVM_ENABLE_ZLIB=FORCE_ON ^
  -DLLVM_ENABLE_ZSTD=FORCE_ON ^
  -DCMAKE_C_FLAGS="%common_compiler_flags%" ^
  -DCMAKE_CXX_FLAGS="%common_compiler_flags%" ^
  -DLLVM_ENABLE_RPMALLOC=ON ^
  -DLLVM_ENABLE_PROJECTS="clang;lld" ^
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" ^
  -DCPACK_GENERATOR="WIX" ^
  -DCOMPILER_RT_BUILD_ORC=OFF

if "%force-msvc%" == "" (
  where /q clang-cl
  if %errorlevel% EQU 0 (
    where /q lld-link
    if %errorlevel% EQU 0 (
      set common_compiler_flags=%common_compiler_flags% -fuse-ld=lld
      
      set common_cmake_flags=%common_cmake_flags%^
        -DCMAKE_C_COMPILER=clang-cl.exe ^
        -DCMAKE_CXX_COMPILER=clang-cl.exe ^
        -DCMAKE_LINKER=lld-link.exe ^
        -DLLVM_ENABLE_LLD=ON ^
        -DCMAKE_C_FLAGS="%common_compiler_flags%" ^
        -DCMAKE_CXX_FLAGS="%common_compiler_flags%"
    )
  )
)

set common_lldb_flags=^
  -DLLDB_RELOCATABLE_PYTHON=1 ^
  -DLLDB_EMBED_PYTHON_HOME=OFF

set cmake_profile_flags=""

REM Preserve original path
set OLDPATH=%PATH%

REM Build the 32-bits and/or 64-bits binaries.
if "%x86%" == "true" call :do_build_32 || exit /b 1
if "%x64%" == "true" call :do_build_64_common amd64 %python64_dir% || exit /b 1
if "%arm64%" == "true" call :do_build_64_common arm64 %pythonarm64_dir% || exit /b 1
exit /b 0

::==============================================================================
:: Build 32-bits binaries.
::==============================================================================
:do_build_32
call :set_environment %python32_dir% || exit /b 1
call "%vsdevcmd%" -arch=x86 || exit /b 1
@echo on
mkdir build32_stage0
cd build32_stage0
call :do_build_libxml || exit /b 1
call :do_build_zlib || exit /b 1
call :do_build_zstd || exit /b 1

REM Stage0 binaries directory; used in stage1.
set "stage0_bin_dir=%build_dir%/build32_stage0/bin"
set cmake_flags=^
  %common_cmake_flags% ^
  -DLLVM_ENABLE_RPMALLOC=OFF ^
  -DPython3_ROOT_DIR=%PYTHONHOME% ^
  -DLIBXML2_INCLUDE_DIR=%libxmldir%/include/libxml2 ^
  -DLIBXML2_LIBRARIES=%libxmldir%/lib/libxml2s.lib ^
  -DZLIB_INCLUDE_DIR=%zlibdir%/include ^
  -DZLIB_LIBRARY=%zlibdir%/lib/zs.lib ^
  -DZLIB_LIBRARY_RELEASE=%zlibdir%/lib/zs.lib ^
  -Dzstd_INCLUDE_DIR=%zstddir%/include ^
  -Dzstd_LIBRARY=%zstddir%/lib/zstd_static.lib

cmake -GNinja %cmake_flags% %llvm_src%\llvm || exit /b 1
ninja || exit /b 1
REM ninja check-llvm || exit /b 1
REM ninja check-clang || exit /b 1
ninja check-lld || exit /b 1
REM ninja check-runtimes || exit /b 1
cd..

REM CMake expects the paths that specifies the compiler and linker to be
REM with forward slash.
set all_cmake_flags=^
  %cmake_flags% ^
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;lldb;" ^
  -DLLVM_ENABLE_RUNTIMES="compiler-rt;openmp" ^
  %common_lldb_flags% ^
  -DPYTHON_HOME=%PYTHONHOME% ^
  -DCMAKE_C_COMPILER=%stage0_bin_dir%/clang-cl.exe ^
  -DCMAKE_CXX_COMPILER=%stage0_bin_dir%/clang-cl.exe ^
  -DCMAKE_LINKER=%stage0_bin_dir%/lld-link.exe ^
  -DCMAKE_AR=%stage0_bin_dir%/llvm-lib.exe ^
  -DCMAKE_RC=%stage0_bin_dir%/llvm-windres.exe
set cmake_flags=%all_cmake_flags:\=/%

mkdir build32
cd build32
cmake -GNinja %cmake_flags% %llvm_src%\llvm || exit /b 1
ninja || exit /b 1
REM ninja check-llvm || exit /b 1
REM ninja check-clang || exit /b 1
ninja check-lld || exit /b 1
REM ninja check-runtimes || exit /b 1
REM ninja check-clang-tools || exit /b 1
ninja package || exit /b 1
cd ..

exit /b 0
::==============================================================================

::==============================================================================
:: Build 64-bits binaries (common function for both x64 and arm64)
::==============================================================================
:do_build_64_common
set arch=%1
set python_dir=%2

call :set_environment %python_dir% || exit /b 1
call "%vsdevcmd%" -arch=%arch% || exit /b 1
@echo on
mkdir build_%arch%_stage0
cd build_%arch%_stage0
call :do_build_libxml || exit /b 1
call :do_build_zlib || exit /b 1
call :do_build_zstd || exit /b 1

REM Stage0 binaries directory; used in stage1.
set "stage0_bin_dir=%build_dir%/build_%arch%_stage0/bin"
set cmake_flags=^
  %common_cmake_flags% ^
  -DPython3_ROOT_DIR=%PYTHONHOME% ^
  -DLIBXML2_INCLUDE_DIR=%libxmldir%/include/libxml2 ^
  -DLIBXML2_LIBRARIES=%libxmldir%/lib/libxml2s.lib ^
  -DZLIB_INCLUDE_DIR=%zlibdir%/include ^
  -DZLIB_LIBRARY=%zlibdir%/lib/zs.lib ^
  -DZLIB_LIBRARY_RELEASE=%zlibdir%/lib/zs.lib ^
  -Dzstd_INCLUDE_DIR=%zstddir%/include ^
  -Dzstd_LIBRARY=%zstddir%/lib/zstd_static.lib ^
  -DCLANG_DEFAULT_LINKER=lld
if "%arch%"=="arm64" (
  set cmake_flags=%cmake_flags% ^
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF
)

cmake -GNinja %cmake_flags% ^
  -DLLVM_TARGETS_TO_BUILD=Native ^
  %llvm_src%\llvm || exit /b 1
ninja clang lld llvm-lib llvm-windres runtimes || exit /b 1
if "%fast-build%" neq "true" (
ninja || exit /b 1
ninja check-llvm || exit /b 1
ninja check-clang || exit /b 1
ninja check-lld || exit /b 1
if "%arch%"=="amd64" (
  ninja check-runtimes || exit /b 1
)
)
cd..

REM CMake expects the paths that specifies the compiler and linker to be
REM with forward slash.
set all_cmake_flags=^
  %cmake_flags% ^
  -DCMAKE_C_COMPILER=%stage0_bin_dir%/clang-cl.exe ^
  -DCMAKE_CXX_COMPILER=%stage0_bin_dir%/clang-cl.exe ^
  -DCMAKE_LINKER=%stage0_bin_dir%/lld-link.exe ^
  -DCMAKE_AR=%stage0_bin_dir%/llvm-lib.exe ^
  -DCMAKE_RC=%stage0_bin_dir%/llvm-windres.exe
if "%arch%"=="arm64" (
  set all_cmake_flags=%all_cmake_flags% ^
    -DCPACK_SYSTEM_NAME=woa64
)
set cmake_flags=%all_cmake_flags:\=/%

mkdir build_%arch%
cd build_%arch%
REM --fast-build skips PGO training for CI speed on time-constrained
REM runners (see build_llvm_release.bat's usage doc), but --enhanced-pgo
REM is an explicit, deliberate opt-in and should not be silently defeated
REM by it; only skip training when enhanced-pgo was not also requested.
if "%fast-build%" == "true" if "%enhanced-pgo%" neq "true" (
  echo Skipping PGO training due to --fast-build.
) else (
  call :do_generate_profile || exit /b 1
)
set lto_cmake_flag=
if "%enable-thinlto%" == "true" set lto_cmake_flag=-DLLVM_ENABLE_LTO=Thin
cmake -GNinja %cmake_flags% ^
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;lldb;flang;mlir" ^
  -DLLVM_ENABLE_RUNTIMES="compiler-rt;openmp" ^
  %lto_cmake_flag% ^
  %common_lldb_flags% ^
  -DPYTHON_HOME=%PYTHONHOME% ^
  %cmake_profile_flags% %llvm_src%\llvm || exit /b 1
ninja || exit /b 1

:: generate tarball with install toolchain only off
if "%arch%"=="amd64" (
  set filename=clang+llvm-%version%-x86_64-pc-windows-msvc
) else (
  set filename=clang+llvm-%version%-aarch64-pc-windows-msvc
)
set main_install_dir=%build_dir%/%filename%
set pdb_install_dir=%build_dir%/%filename%-pdb-root
REM LLVM_ENABLE_PDB flips the /Zi compile flag, which invalidates every
REM object file from the build above and forces a full recompile+relink.
REM That recompile does not depend on the test suite or WiX packaging
REM below, so kick it off now in a separate build directory and let it
REM run concurrently with them, hiding most/all of its wall-clock cost
REM instead of paying for it serially at the end. LLVM_ENABLE_PDB is
REM still never set for the MSI/WiX "ninja package" build below:
REM bundling PDBs into the WiX-generated MSI causes CPack/WiX packaging
REM failures, so PDBs are packaged separately as their own tarball
REM instead.
if "%enable-pdb%" == "true" (
  cd ..
  mkdir build_%arch%_pdb
  cd build_%arch%_pdb
  cmake -GNinja %cmake_flags% ^
    -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;lldb;flang;mlir" ^
    -DLLVM_ENABLE_RUNTIMES="compiler-rt;openmp" ^
    %lto_cmake_flag% ^
    %common_lldb_flags% ^
    -DPYTHON_HOME=%PYTHONHOME% ^
    %cmake_profile_flags% -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF ^
    -DCMAKE_INSTALL_PREFIX=%pdb_install_dir% ^
    -DLLVM_ENABLE_PDB=ON ^
    %llvm_src%\llvm || exit /b 1
  del /q ..\pdb_build.done 2>nul
  REM Use "start /min" (a genuinely separate, minimized console) rather
  REM than "start /b" (which shares the parent's console/IO handles):
  REM the background job's own console output can otherwise bleed into
  REM and interleave with this foreground script's captured output
  REM despite the ">" file redirection below, consistent with
  REM console-handle contention between the two concurrently-running
  REM processes. A dedicated console avoids sharing those handles at all.
  REM "/v:on" + "^!errorlevel^!" (rather than "%%errorlevel%%") is
  REM required so ninja's real exit code is captured. Without delayed
  REM expansion, cmd.exe substitutes %errorlevel% once when the whole
  REM "cmd1 & cmd2" line is parsed (before ninja even runs), so it would
  REM always report the pre-existing errorlevel instead of ninja's
  REM result.
  start "pdb_build" /min cmd /v:on /c "ninja install > ..\pdb_build.log 2>&1 & echo ^!errorlevel^! > ..\pdb_build.done" < nul
  cd ..\build_%arch%
)
ninja check-llvm || exit /b 1
ninja check-clang || exit /b 1
ninja check-lld || exit /b 1
if "%arch%"=="amd64" (
  ninja check-runtimes || exit /b 1
)
ninja check-clang-tools || exit /b 1
ninja check-clangd || exit /b 1
REM ninja check-flang || exit /b 1
REM ninja check-mlir || exit /b 1
REM ninja check-lldb || exit /b 1
ninja package || exit /b 1

if "%enable-pdb%" == "true" (
  call :wait_for_pdb_build || exit /b 1
)
cmake -GNinja %cmake_flags% %cmake_profile_flags% -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF ^
  -DCMAKE_INSTALL_PREFIX=%main_install_dir% ^
  %llvm_src%\llvm || exit /b 1
ninja install || exit /b 1
:: check llvm_config is present & returns something
%main_install_dir%/bin/llvm-config.exe --bindir || exit /b 1
cd ..
if "%enable-pdb%" == "true" (
  :: Package PDBs from the separate PDB-enabled install tree so the main
  :: archive can still come from the clean non-PDB install tree above.
  set pdb_filename=%filename%-pdb
  REM Use a plain relative directory name here, not the absolute
  REM %pdb_install_dir% path: on this runner "pushd" fails with
  REM "The system cannot find the drive specified" when given an
  REM absolute path that mixes backslash and forward-slash separators
  REM (as %pdb_install_dir% does, since it's built with a "/" against
  REM the backslash-based %build_dir%). cwd is already %build_dir%
  REM (see "cd .." above), and %filename%-pdb-root is a direct child
  REM of it, so the relative form below is equivalent and avoids the
  REM issue entirely.
  pushd %filename%-pdb-root
  7z a -ttar -so ..\!pdb_filename!.tar bin\*.pdb lib\*.pdb | 7z a -txz -si ..\!pdb_filename!.tar.xz
  popd
)
7z a -ttar -so %filename%.tar %filename% | 7z a -txz -si %filename%.tar.xz

exit /b 0

::==============================================================================
:: Poll for the concurrent PDB build kicked off earlier in this function to
:: finish, and propagate its exit code. Must be a standalone function (not
:: an inline goto/label inside a parenthesized if-block) since cmd.exe does
:: not reliably support jumping to a label defined inside the same
:: "( ... )" block.
::==============================================================================
:wait_for_pdb_build
if not exist ..\pdb_build.done (
  ping -n 6 127.0.0.1 >nul
  goto :wait_for_pdb_build
)
REM Use "for /f" rather than "set /p" to read the exit code: "for /f"
REM tokenizes on whitespace and strips it, whereas "set /p" would take any
REM trailing spaces/junk in the file literally, breaking the "== 0" check
REM below even when the underlying build actually succeeded.
set pdb_build_rc=
for /f %%r in (..\pdb_build.done) do set pdb_build_rc=%%r
if not "%pdb_build_rc%" == "0" (
  type ..\pdb_build.log
  exit /b 1
)
exit /b 0

::==============================================================================
:: Set PATH and some environment variables.
::==============================================================================
:set_environment
REM Restore original path
set PATH=%OLDPATH%

set python_dir=%1

REM Set Python environment
if "%local-python%" == "true" (
  FOR /F "delims=" %%i IN ('where python.exe ^| head -1') DO set python_exe=%%i
  set PYTHONHOME=!python_exe:~0,-11!
) else (
  %python_dir%/python.exe --version || exit /b 1
  set PYTHONHOME=%python_dir%
)
set PATH=%PYTHONHOME%;%PATH%

set "VSCMD_START_DIR=%build_dir%"

exit /b 0

::=============================================================================

::==============================================================================
:: Verify checksum.
::==============================================================================
:verify_checksum
cmake -E %2sum %1 > %1.%2sum
echo %3  %1> %1.%2sum.orig
cmake -E compare_files --ignore-eol %1.%2sum %1.%2sum.orig
if %ERRORLEVEL% NEQ 0 (
  echo verify_checksum failed for %1
  exit /b 1
)
exit /b 0

::==============================================================================
:: Build libxml.
::==============================================================================
:do_build_libxml
mkdir libxmlbuild
cd libxmlbuild
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install ^
  -DBUILD_SHARED_LIBS=OFF -DLIBXML2_WITH_C14N=OFF -DLIBXML2_WITH_CATALOG=OFF ^
  -DLIBXML2_WITH_DEBUG=OFF -DLIBXML2_WITH_DOCB=OFF -DLIBXML2_WITH_FTP=OFF ^
  -DLIBXML2_WITH_HTML=OFF -DLIBXML2_WITH_HTTP=OFF -DLIBXML2_WITH_ICONV=OFF ^
  -DLIBXML2_WITH_ICU=OFF -DLIBXML2_WITH_ISO8859X=OFF -DLIBXML2_WITH_LEGACY=OFF ^
  -DLIBXML2_WITH_LZMA=OFF -DLIBXML2_WITH_MEM_DEBUG=OFF -DLIBXML2_WITH_MODULES=OFF ^
  -DLIBXML2_WITH_OUTPUT=ON -DLIBXML2_WITH_PATTERN=OFF -DLIBXML2_WITH_PROGRAMS=OFF ^
  -DLIBXML2_WITH_PUSH=OFF -DLIBXML2_WITH_PYTHON=OFF -DLIBXML2_WITH_READER=OFF ^
  -DLIBXML2_WITH_REGEXPS=OFF -DLIBXML2_WITH_RUN_DEBUG=OFF -DLIBXML2_WITH_SAX1=ON ^
  -DLIBXML2_WITH_SCHEMAS=OFF -DLIBXML2_WITH_SCHEMATRON=OFF -DLIBXML2_WITH_TESTS=OFF ^
  -DLIBXML2_WITH_THREADS=ON -DLIBXML2_WITH_THREAD_ALLOC=OFF -DLIBXML2_WITH_TREE=ON ^
  -DLIBXML2_WITH_VALID=OFF -DLIBXML2_WITH_WRITER=OFF -DLIBXML2_WITH_XINCLUDE=OFF ^
  -DLIBXML2_WITH_XPATH=OFF -DLIBXML2_WITH_XPTR=OFF -DLIBXML2_WITH_ZLIB=OFF ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  ../../libxml2-v%libxml_version% || exit /b 1
ninja install || exit /b 1
set libxmldir=%cd%\install
set "libxmldir=%libxmldir:\=/%"
cd ..
exit /b 0

::==============================================================================
:: Build zlib.
::==============================================================================
:do_build_zlib
mkdir zlibbuild
cd zlibbuild
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install ^
  -DZLIB_BUILD_TESTING=OFF -DZLIB_BUILD_SHARED=OFF -DZLIB_BUILD_STATIC=ON ^
  -DZLIB_INSTALL=ON -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  ../../zlib-%zlib_version% || exit /b 1
ninja install || exit /b 1
set zlibdir=%cd%\install
set "zlibdir=%zlibdir:\=/%"
cd ..
exit /b 0

::==============================================================================
:: Build zstd.
::==============================================================================
:do_build_zstd
mkdir zstdbuild
cd zstdbuild
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install ^
  -DZSTD_BUILD_PROGRAMS=ON -DZSTD_BUILD_TESTS=OFF -DZSTD_BUILD_STATIC=ON ^
  -DZSTD_BUILD_SHARED=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  ../../zstd-%zstd_version%/build/cmake || exit /b 1
ninja install || exit /b 1
set zstddir=%cd%\install
set "zstddir=%zstddir:\=/%"
cd ..
exit /b 0

::==============================================================================
:: Generate a PGO profile.
::==============================================================================
:do_generate_profile
REM Build Clang with instrumentation.
mkdir instrument
cd instrument
cmake -GNinja %cmake_flags% -DLLVM_TARGETS_TO_BUILD=Native ^
  -DLLVM_BUILD_INSTRUMENTED=IR %llvm_src%\llvm || exit /b 1
ninja clang || exit /b 1
set instrumented_clang=%cd:\=/%/bin/clang-cl.exe
cd ..
mkdir train
cd train
if "%enhanced-pgo%" == "true" (
  REM Build LLVMSupport with the instrumented clang to generate a broad profile.
  REM This mirrors Linux and Mac perf-training approach.
  cmake -GNinja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=%instrumented_clang% ^
    -DCMAKE_CXX_COMPILER=%instrumented_clang% ^
    -DLLVM_TARGETS_TO_BUILD=Native ^
    -DLLVM_ENABLE_PROJECTS="" ^
    -DLLVM_ENABLE_RUNTIMES="" ^
    %llvm_src%\llvm || exit /b 1
  REM Drop profiles generated from running cmake; those are not representative.
  del ..\instrument\profiles\*.profraw
  ninja LLVMSupport || exit /b 1
) else (
  REM Use instrumented build of clang to compile a complex single file to
  REM deliver minimum build times.
  cmake -GNinja %cmake_flags% ^
    -DCMAKE_C_COMPILER=%instrumented_clang% ^
    -DCMAKE_CXX_COMPILER=%instrumented_clang% ^
    -DLLVM_ENABLE_PROJECTS=clang ^
    -DLLVM_TARGETS_TO_BUILD=Native ^
    %llvm_src%\llvm || exit /b 1
  REM Drop profiles generated from running cmake; those are not representative.
  del ..\instrument\profiles\*.profraw
  ninja tools/clang/lib/Sema/CMakeFiles/obj.clangSema.dir/Sema.cpp.obj || exit /b 1
)
cd ..
set profile=%cd:\=/%/profile.profdata
%stage0_bin_dir%\llvm-profdata merge -output=%profile% instrument\profiles\*.profraw || exit /b 1
set common_compiler_flags=%common_compiler_flags% -Wno-backend-plugin
set cmake_profile_flags=-DLLVM_PROFDATA_FILE=%profile% ^
  -DCMAKE_C_FLAGS="%common_compiler_flags%" ^
  -DCMAKE_CXX_FLAGS="%common_compiler_flags%"
exit /b 0

::=============================================================================
:: Parse command line arguments.
:: The format for the arguments is:
::   Boolean: --option
::   Value:   --option<separator>value
::     with <separator> being: space, colon, semicolon or equal sign
::
:: Command line usage example:
::   my-batch-file.bat --build --type=release --version 123
:: It will create 3 variables:
::   'build' with the value 'true'
::   'type' with the value 'release'
::   'version' with the value '123'
::
:: Usage:
::   set "build="
::   set "type="
::   set "version="
::
::   REM Parse arguments.
::   call :parse_args %*
::
::   if defined build (
::     ...
::   )
::   if %type%=='release' (
::     ...
::   )
::   if %version%=='123' (
::     ...
::   )
::=============================================================================
:parse_args
  set "arg_name="
  :parse_args_start
  if "%1" == "" (
    :: Set a seen boolean argument.
    if "%arg_name%" neq "" (
      set "%arg_name%=true"
    )
    goto :parse_args_done
  )
  set aux=%1
  if "%aux:~0,2%" == "--" (
    :: Set a seen boolean argument.
    if "%arg_name%" neq "" (
      set "%arg_name%=true"
    )
    set "arg_name=%aux:~2,250%"
  ) else (
    set "%arg_name%=%1"
    set "arg_name="
  )
  shift
  goto :parse_args_start

:parse_args_done
exit /b 0
