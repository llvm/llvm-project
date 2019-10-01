#!/usr/bin/env bash
#===-- test-release.sh - Test the LLVM release candidates ------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# Download, build, and test the release candidate for an LLVM release.
#
#===------------------------------------------------------------------------===#

System=`uname -s`
if [ "$System" = "FreeBSD" ]; then
    MAKE=gmake
else
    MAKE=make
fi
generator="Unix Makefiles"

# Base SVN URL for the sources.
Base_url="http://llvm.org/svn/llvm-project"

Release=""
Release_no_dot=""
RC=""
Triple=""
use_gzip="no"
do_checkout="yes"
do_debug="no"
do_asserts="no"
do_compare="yes"
do_rt="yes"
do_libs="yes"
do_libcxxabi="yes"
do_libunwind="yes"
do_test_suite="yes"
do_openmp="yes"
do_lld="yes"
do_lldb="no"
do_polly="yes"
BuildDir="`pwd`"
ExtraConfigureFlags=""
ExportBranch=""

function usage() {
    echo "usage: `basename $0` -release X.Y.Z -rc NUM [OPTIONS]"
    echo ""
    echo " -release X.Y.Z       The release version to test."
    echo " -rc NUM              The pre-release candidate number."
    echo " -final               The final release candidate."
    echo " -triple TRIPLE       The target triple for this machine."
    echo " -j NUM               Number of compile jobs to run. [default: 3]"
    echo " -build-dir DIR       Directory to perform testing in. [default: pwd]"
    echo " -no-checkout         Don't checkout the sources from SVN."
    echo " -test-debug          Test the debug build. [default: no]"
    echo " -test-asserts        Test with asserts on. [default: no]"
    echo " -no-compare-files    Don't test that phase 2 and 3 files are identical."
    echo " -use-gzip            Use gzip instead of xz."
    echo " -use-ninja           Use ninja instead of make/gmake."
    echo " -configure-flags FLAGS  Extra flags to pass to the configure step."
    echo " -svn-path DIR        Use the specified DIR instead of a release."
    echo "                      For example -svn-path trunk or -svn-path branches/release_37"
    echo " -no-rt               Disable check-out & build Compiler-RT"
    echo " -no-libs             Disable check-out & build libcxx/libcxxabi/libunwind"
    echo " -no-libcxxabi        Disable check-out & build libcxxabi"
    echo " -no-libunwind        Disable check-out & build libunwind"
    echo " -no-test-suite       Disable check-out & build test-suite"
    echo " -no-openmp           Disable check-out & build libomp"
    echo " -no-lld              Disable check-out & build lld"
    echo " -lldb                Enable check-out & build lldb"
    echo " -no-lldb             Disable check-out & build lldb (default)"
    echo " -no-polly            Disable check-out & build Polly"
}

while [ $# -gt 0 ]; do
    case $1 in
        -release | --release )
            shift
            Release="$1"
            Release_no_dot="`echo $1 | sed -e 's,\.,,g'`"
            ;;
        -rc | --rc | -RC | --RC )
            shift
            RC="rc$1"
            ;;
        -final | --final )
            RC=final
            ;;
        -svn-path | --svn-path )
            shift
            Release="test"
            Release_no_dot="test"
            ExportBranch="$1"
            RC="`echo $ExportBranch | sed -e 's,/,_,g'`"
            echo "WARNING: Using the branch $ExportBranch instead of a release tag"
            echo "         This is intended to aid new packagers in trialing "
            echo "         builds without requiring a tag to be created first"
            ;;
        -triple | --triple )
            shift
            Triple="$1"
            ;;
        -configure-flags | --configure-flags )
            shift
            ExtraConfigureFlags="$1"
            ;;
        -j* )
            NumJobs="`echo $1 | sed -e 's,-j\([0-9]*\),\1,g'`"
            if [ -z "$NumJobs" ]; then
                shift
                NumJobs="$1"
            fi
            ;;
        -use-ninja )
            MAKE=ninja
            generator=Ninja
            ;;
        -build-dir | --build-dir | -builddir | --builddir )
            shift
            BuildDir="$1"
            ;;
        -no-checkout | --no-checkout )
            do_checkout="no"
            ;;
        -test-debug | --test-debug )
            do_debug="yes"
            ;;
        -test-asserts | --test-asserts )
            do_asserts="yes"
            ;;
        -no-compare-files | --no-compare-files )
            do_compare="no"
            ;;
        -use-gzip | --use-gzip )
            use_gzip="yes"
            ;;
        -no-rt )
            do_rt="no"
            ;;
        -no-libs )
            do_libs="no"
            ;;
        -no-libcxxabi )
            do_libcxxabi="no"
            ;;
        -no-libunwind )
            do_libunwind="no"
            ;;
        -no-test-suite )
            do_test_suite="no"
            ;;
        -no-openmp )
            do_openmp="no"
            ;;
        -no-lld )
            do_lld="no"
            ;;
        -lldb )
            do_lldb="yes"
            ;;
        -no-lldb )
            do_lldb="no"
            ;;
        -no-polly )
            do_polly="no"
            ;;
        -help | --help | -h | --h | -\? )
            usage
            exit 0
            ;;
        * )
            echo "unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# Check required arguments.
if [ -z "$Release" ]; then
    echo "error: no release number specified"
    exit 1
fi
if [ -z "$RC" ]; then
    echo "error: no release candidate number specified"
    exit 1
fi
if [ -z "$ExportBranch" ]; then
    ExportBranch="tags/RELEASE_$Release_no_dot/$RC"
fi
if [ -z "$Triple" ]; then
    echo "error: no target triple specified"
    exit 1
fi

# Figure out how many make processes to run.
if [ -z "$NumJobs" ]; then
    NumJobs=`sysctl -n hw.activecpu 2> /dev/null || true`
fi
if [ -z "$NumJobs" ]; then
    NumJobs=`sysctl -n hw.ncpu 2> /dev/null || true`
fi
if [ -z "$NumJobs" ]; then
    NumJobs=`grep -c processor /proc/cpuinfo 2> /dev/null || true`
fi
if [ -z "$NumJobs" ]; then
    NumJobs=3
fi

# Projects list
projects="llvm cfe clang-tools-extra"
if [ $do_rt = "yes" ]; then
  projects="$projects compiler-rt"
fi
if [ $do_libs = "yes" ]; then
  projects="$projects libcxx"
  if [ $do_libcxxabi = "yes" ]; then
    projects="$projects libcxxabi"
  fi
  if [ $do_libunwind = "yes" ]; then
    projects="$projects libunwind"
  fi
fi
case $do_test_suite in
  yes|export-only)
    projects="$projects test-suite"
    ;;
esac
if [ $do_openmp = "yes" ]; then
  projects="$projects openmp"
fi
if [ $do_lld = "yes" ]; then
  projects="$projects lld"
fi
if [ $do_lldb = "yes" ]; then
  projects="$projects lldb"
fi
if [ $do_polly = "yes" ]; then
  projects="$projects polly"
fi

# Go to the build directory (may be different from CWD)
BuildDir=$BuildDir/$RC
mkdir -p $BuildDir
cd $BuildDir

# Location of log files.
LogDir=$BuildDir/logs
mkdir -p $LogDir

# Final package name.
Package=clang+llvm-$Release
if [ $RC != "final" ]; then
  Package=$Package-$RC
fi
Package=$Package-$Triple

# Errors to be highlighted at the end are written to this file.
echo -n > $LogDir/deferred_errors.log

function deferred_error() {
  Phase="$1"
  Flavor="$2"
  Msg="$3"
  echo "[${Flavor} Phase${Phase}] ${Msg}" | tee -a $LogDir/deferred_errors.log
}

# Make sure that a required program is available
function check_program_exists() {
  local program="$1"
  if ! type -P $program > /dev/null 2>&1 ; then
    echo "program '$1' not found !"
    exit 1
  fi
}

if [ "$System" != "Darwin" ]; then
  check_program_exists 'chrpath'
  check_program_exists 'file'
  check_program_exists 'objdump'
fi

check_program_exists ${MAKE}

# Make sure that the URLs are valid.
function check_valid_urls() {
    for proj in $projects ; do
        echo "# Validating $proj SVN URL"

        if ! svn ls $Base_url/$proj/$ExportBranch > /dev/null 2>&1 ; then
            echo "$proj does not have a $ExportBranch branch/tag!"
            exit 1
        fi
    done
}

# Export sources to the build directory.
function export_sources() {
    check_valid_urls

    for proj in $projects ; do
        case $proj in
        llvm)
            projsrc=$proj.src
            ;;
        cfe)
            projsrc=llvm.src/tools/clang
            ;;
        lld|lldb|polly)
            projsrc=llvm.src/tools/$proj
            ;;
        clang-tools-extra)
            projsrc=llvm.src/tools/clang/tools/extra
            ;;
        compiler-rt|libcxx|libcxxabi|libunwind|openmp)
            projsrc=llvm.src/projects/$proj
            ;;
        test-suite)
            projsrc=$proj.src
            ;;
        *)
            echo "error: unknown project $proj"
            exit 1
            ;;
        esac

        if [ -d $projsrc ]; then
          echo "# Reusing $proj $Release-$RC sources in $projsrc"
          continue
        fi
        echo "# Exporting $proj $Release-$RC sources to $projsrc"
        if ! svn export -q $Base_url/$proj/$ExportBranch $projsrc ; then
            echo "error: failed to export $proj project"
            exit 1
        fi
    done

    cd $BuildDir
}

function configure_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"

    case $Flavor in
        Release )
            BuildType="Release"
            Assertions="OFF"
            ;;
        Release+Asserts )
            BuildType="Release"
            Assertions="ON"
            ;;
        Debug )
            BuildType="Debug"
            Assertions="ON"
            ;;
        * )
            echo "# Invalid flavor '$Flavor'"
            echo ""
            return
            ;;
    esac

    echo "# Using C compiler: $c_compiler"
    echo "# Using C++ compiler: $cxx_compiler"

    cd $ObjDir
    echo "# Configuring llvm $Release-$RC $Flavor"

    echo "#" env CC="$c_compiler" CXX="$cxx_compiler" \
        cmake -G "$generator" \
        -DCMAKE_BUILD_TYPE=$BuildType -DLLVM_ENABLE_ASSERTIONS=$Assertions \
        $ExtraConfigureFlags $BuildDir/llvm.src \
        2>&1 | tee $LogDir/llvm.configure-Phase$Phase-$Flavor.log
    env CC="$c_compiler" CXX="$cxx_compiler" \
        cmake -G "$generator" \
        -DCMAKE_BUILD_TYPE=$BuildType -DLLVM_ENABLE_ASSERTIONS=$Assertions \
        $ExtraConfigureFlags $BuildDir/llvm.src \
        2>&1 | tee $LogDir/llvm.configure-Phase$Phase-$Flavor.log

    cd $BuildDir
}

function build_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"
    DestDir="$4"

    Verbose="VERBOSE=1"
    if [ ${MAKE} = 'ninja' ]; then
      Verbose="-v"
    fi

    cd $ObjDir
    echo "# Compiling llvm $Release-$RC $Flavor"
    echo "# ${MAKE} -j $NumJobs $Verbose"
    ${MAKE} -j $NumJobs $Verbose \
        2>&1 | tee $LogDir/llvm.make-Phase$Phase-$Flavor.log

    echo "# Installing llvm $Release-$RC $Flavor"
    echo "# ${MAKE} install"
    DESTDIR="${DestDir}" ${MAKE} install \
        2>&1 | tee $LogDir/llvm.install-Phase$Phase-$Flavor.log
    cd $BuildDir
}

function test_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"

    KeepGoing="-k"
    if [ ${MAKE} = 'ninja' ]; then
      # Ninja doesn't have a documented "keep-going-forever" mode, we need to
      # set a limit on how many jobs can fail before we give up.
      KeepGoing="-k 100"
    fi

    cd $ObjDir
    if ! ( ${MAKE} -j $NumJobs $KeepGoing check-all \
        2>&1 | tee $LogDir/llvm.check-Phase$Phase-$Flavor.log ) ; then
      deferred_error $Phase $Flavor "check-all failed"
    fi

    if [ $do_test_suite = 'yes' ]; then
      cd $TestSuiteBuildDir
      env CC="$c_compiler" CXX="$cxx_compiler" \
          cmake $TestSuiteSrcDir -G "$generator" -DTEST_SUITE_LIT=$Lit

      if ! ( ${MAKE} -j $NumJobs $KeepGoing check \
          2>&1 | tee $LogDir/llvm.check-Phase$Phase-$Flavor.log ) ; then
        deferred_error $Phase $Flavor "test suite failed"
      fi
    fi
    cd $BuildDir
}

# Clean RPATH. Libtool adds the build directory to the search path, which is
# not necessary --- and even harmful --- for the binary packages we release.
function clean_RPATH() {
  if [ "$System" = "Darwin" ]; then
    return
  fi
  local InstallPath="$1"
  for Candidate in `find $InstallPath/{bin,lib} -type f`; do
    if file $Candidate | grep ELF | egrep 'executable|shared object' > /dev/null 2>&1 ; then
      if rpath=`objdump -x $Candidate | grep 'RPATH'` ; then
        rpath=`echo $rpath | sed -e's/^ *RPATH *//'`
        if [ -n "$rpath" ]; then
          newrpath=`echo $rpath | sed -e's/.*\(\$ORIGIN[^:]*\).*/\1/'`
          chrpath -r $newrpath $Candidate 2>&1 > /dev/null 2>&1
        fi
      fi
    fi
  done
}

# Create a package of the release binaries.
function package_release() {
    cwd=`pwd`
    cd $BuildDir/Phase3/Release
    mv llvmCore-$Release-$RC.install/usr/local $Package
    if [ "$use_gzip" = "yes" ]; then
      tar cfz $BuildDir/$Package.tar.gz $Package
    else
      tar cfJ $BuildDir/$Package.tar.xz $Package
    fi
    mv $Package llvmCore-$Release-$RC.install/usr/local
    cd $cwd
}

# Exit if any command fails
# Note: pipefail is necessary for running build commands through
# a pipe (i.e. it changes the output of ``false | tee /dev/null ; echo $?``)
set -e
set -o pipefail

if [ "$do_checkout" = "yes" ]; then
    export_sources
fi

# Setup the test-suite.  Do this early so we can catch failures before
# we do the full 3 stage build.
if [ $do_test_suite = "yes" ]; then
  SandboxDir="$BuildDir/sandbox"
  Lit=$SandboxDir/bin/lit
  TestSuiteBuildDir="$BuildDir/test-suite-build"
  TestSuiteSrcDir="$BuildDir/test-suite.src"

  virtualenv $SandboxDir
  $SandboxDir/bin/python $BuildDir/llvm.src/utils/lit/setup.py install
  mkdir -p $TestSuiteBuildDir
fi

(
Flavors="Release"
if [ "$do_debug" = "yes" ]; then
    Flavors="Debug $Flavors"
fi
if [ "$do_asserts" = "yes" ]; then
    Flavors="$Flavors Release+Asserts"
fi

for Flavor in $Flavors ; do
    echo ""
    echo ""
    echo "********************************************************************************"
    echo "  Release:     $Release-$RC"
    echo "  Build:       $Flavor"
    echo "  System Info: "
    echo "    `uname -a`"
    echo "********************************************************************************"
    echo ""

    c_compiler="$CC"
    cxx_compiler="$CXX"
    llvmCore_phase1_objdir=$BuildDir/Phase1/$Flavor/llvmCore-$Release-$RC.obj
    llvmCore_phase1_destdir=$BuildDir/Phase1/$Flavor/llvmCore-$Release-$RC.install

    llvmCore_phase2_objdir=$BuildDir/Phase2/$Flavor/llvmCore-$Release-$RC.obj
    llvmCore_phase2_destdir=$BuildDir/Phase2/$Flavor/llvmCore-$Release-$RC.install

    llvmCore_phase3_objdir=$BuildDir/Phase3/$Flavor/llvmCore-$Release-$RC.obj
    llvmCore_phase3_destdir=$BuildDir/Phase3/$Flavor/llvmCore-$Release-$RC.install

    rm -rf $llvmCore_phase1_objdir
    rm -rf $llvmCore_phase1_destdir

    rm -rf $llvmCore_phase2_objdir
    rm -rf $llvmCore_phase2_destdir

    rm -rf $llvmCore_phase3_objdir
    rm -rf $llvmCore_phase3_destdir

    mkdir -p $llvmCore_phase1_objdir
    mkdir -p $llvmCore_phase1_destdir

    mkdir -p $llvmCore_phase2_objdir
    mkdir -p $llvmCore_phase2_destdir

    mkdir -p $llvmCore_phase3_objdir
    mkdir -p $llvmCore_phase3_destdir

    ############################################################################
    # Phase 1: Build llvmCore and clang
    echo "# Phase 1: Building llvmCore"
    configure_llvmCore 1 $Flavor $llvmCore_phase1_objdir
    build_llvmCore 1 $Flavor \
        $llvmCore_phase1_objdir $llvmCore_phase1_destdir
    clean_RPATH $llvmCore_phase1_destdir/usr/local

    ########################################################################
    # Phase 2: Build llvmCore with newly built clang from phase 1.
    c_compiler=$llvmCore_phase1_destdir/usr/local/bin/clang
    cxx_compiler=$llvmCore_phase1_destdir/usr/local/bin/clang++
    echo "# Phase 2: Building llvmCore"
    configure_llvmCore 2 $Flavor $llvmCore_phase2_objdir
    build_llvmCore 2 $Flavor \
        $llvmCore_phase2_objdir $llvmCore_phase2_destdir
    clean_RPATH $llvmCore_phase2_destdir/usr/local

    ########################################################################
    # Phase 3: Build llvmCore with newly built clang from phase 2.
    c_compiler=$llvmCore_phase2_destdir/usr/local/bin/clang
    cxx_compiler=$llvmCore_phase2_destdir/usr/local/bin/clang++
    echo "# Phase 3: Building llvmCore"
    configure_llvmCore 3 $Flavor $llvmCore_phase3_objdir
    build_llvmCore 3 $Flavor \
        $llvmCore_phase3_objdir $llvmCore_phase3_destdir
    clean_RPATH $llvmCore_phase3_destdir/usr/local

    ########################################################################
    # Testing: Test phase 3
    c_compiler=$llvmCore_phase3_destdir/usr/local/bin/clang
    cxx_compiler=$llvmCore_phase3_destdir/usr/local/bin/clang++
    echo "# Testing - built with clang"
    test_llvmCore 3 $Flavor $llvmCore_phase3_objdir

    ########################################################################
    # Compare .o files between Phase2 and Phase3 and report which ones
    # differ.
    if [ "$do_compare" = "yes" ]; then
        echo
        echo "# Comparing Phase 2 and Phase 3 files"
        for p2 in `find $llvmCore_phase2_objdir -name '*.o'` ; do
            p3=`echo $p2 | sed -e 's,Phase2,Phase3,'`
            # Substitute 'Phase2' for 'Phase3' in the Phase 2 object file in
            # case there are build paths in the debug info. On some systems,
            # sed adds a newline to the output, so pass $p3 through sed too.
            if ! cmp -s \
                <(env LC_CTYPE=C sed -e 's,Phase2,Phase3,g' -e 's,Phase1,Phase2,g' $p2) \
                <(env LC_CTYPE=C sed -e '' $p3) 16 16; then
                echo "file `basename $p2` differs between phase 2 and phase 3"
            fi
        done
    fi
done

) 2>&1 | tee $LogDir/testing.$Release-$RC.log

if [ "$use_gzip" = "yes" ]; then
  echo "# Packaging the release as $Package.tar.gz"
else
  echo "# Packaging the release as $Package.tar.xz"
fi
package_release

set +e

# Woo hoo!
echo "### Testing Finished ###"
echo "### Logs: $LogDir"

echo "### Errors:"
if [ -s "$LogDir/deferred_errors.log" ]; then
  cat "$LogDir/deferred_errors.log"
  exit 1
else
  echo "None."
fi

exit 0
