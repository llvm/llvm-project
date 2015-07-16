#!/usr/bin/env bash
#===-- test-release.sh - Test the LLVM release candidates ------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License.
#
#===------------------------------------------------------------------------===#
#
# Download, build, and test the release candidate for an LLVM release.
#
#===------------------------------------------------------------------------===#

if [ `uname -s` = "FreeBSD" ]; then
    MAKE=gmake
else
    MAKE=make
fi

projects="llvm cfe compiler-rt libcxx libcxxabi test-suite clang-tools-extra"

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
BuildDir="`pwd`"
use_autoconf="no"
ExtraConfigureFlags=""

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
    echo " -configure-flags FLAGS  Extra flags to pass to the configure step."
    echo " -use-autoconf        Use autoconf instead of cmake"
}

if [ `uname -s` = "Darwin" ]; then
  # compiler-rt doesn't yet build with CMake on Darwin.
  use_autoconf="yes"
fi

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
        -use-autoconf | --use-autoconf )
            use_autoconf="yes"
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

# Make sure that a required program is available
function check_program_exists() {
  local program="$1"
  if ! type -P $program > /dev/null 2>&1 ; then
    echo "program '$1' not found !"
    exit 1
  fi
}

if [ `uname -s` != "Darwin" ]; then
  check_program_exists 'chrpath'
  check_program_exists 'file'
  check_program_exists 'objdump'
fi

# Make sure that the URLs are valid.
function check_valid_urls() {
    for proj in $projects ; do
        echo "# Validating $proj SVN URL"

        if ! svn ls $Base_url/$proj/tags/RELEASE_$Release_no_dot/$RC > /dev/null 2>&1 ; then
            echo "$proj $Release release candidate $RC doesn't exist!"
            exit 1
        fi
    done
}

# Export sources to the build directory.
function export_sources() {
    check_valid_urls

    for proj in $projects ; do
        echo "# Exporting $proj $Release-$RC sources"
        if ! svn export -q $Base_url/$proj/tags/RELEASE_$Release_no_dot/$RC $proj.src ; then
            echo "error: failed to export $proj project"
            exit 1
        fi
    done

    echo "# Creating symlinks"
    cd $BuildDir/llvm.src/tools
    if [ ! -h clang ]; then
        ln -s ../../cfe.src clang
    fi
    cd $BuildDir/llvm.src/tools/clang/tools
    if [ ! -h clang-tools-extra ]; then
        ln -s ../../../../clang-tools-extra.src extra
    fi
    cd $BuildDir/llvm.src/projects
    if [ ! -h test-suite ]; then
        ln -s ../../test-suite.src test-suite
    fi
    if [ ! -h compiler-rt ]; then
        ln -s ../../compiler-rt.src compiler-rt
    fi
    if [ ! -h libcxx ]; then
        ln -s ../../libcxx.src libcxx
    fi
    if [ ! -h libcxxabi ]; then
        ln -s ../../libcxxabi.src libcxxabi
    fi
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
            ConfigureFlags="--enable-optimized --disable-assertions"
            ;;
        Release+Asserts )
            BuildType="Release"
            Assertions="ON"
            ConfigureFlags="--enable-optimized --enable-assertions"
            ;;
        Debug )
            BuildType="Debug"
            Assertions="ON"
            ConfigureFlags="--disable-optimized --enable-assertions"
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

    if [ "$use_autoconf" = "yes" ]; then
        echo "#" env CC="$c_compiler" CXX="$cxx_compiler" \
            $BuildDir/llvm.src/configure \
            $ConfigureFlags --disable-timestamps $ExtraConfigureFlags \
            2>&1 | tee $LogDir/llvm.configure-Phase$Phase-$Flavor.log
        env CC="$c_compiler" CXX="$cxx_compiler" \
            $BuildDir/llvm.src/configure \
            $ConfigureFlags --disable-timestamps $ExtraConfigureFlags \
            2>&1 | tee $LogDir/llvm.configure-Phase$Phase-$Flavor.log
    else
        echo "#" env CC="$c_compiler" CXX="$cxx_compiler" \
            cmake -G "Unix Makefiles" \
            -DCMAKE_BUILD_TYPE=$BuildType -DLLVM_ENABLE_ASSERTIONS=$Assertions \
            -DLLVM_ENABLE_TIMESTAMPS=OFF -DLLVM_CONFIGTIME="(timestamp not enabled)" \
            $ExtraConfigureFlags $BuildDir/llvm.src \
            2>&1 | tee $LogDir/llvm.configure-Phase$Phase-$Flavor.log
        env CC="$c_compiler" CXX="$cxx_compiler" \
            cmake -G "Unix Makefiles" \
            -DCMAKE_BUILD_TYPE=$BuildType -DLLVM_ENABLE_ASSERTIONS=$Assertions \
            -DLLVM_ENABLE_TIMESTAMPS=OFF -DLLVM_CONFIGTIME="(timestamp not enabled)" \
            $ExtraConfigureFlags $BuildDir/llvm.src \
            2>&1 | tee $LogDir/llvm.configure-Phase$Phase-$Flavor.log
    fi

    cd $BuildDir
}

function build_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"
    DestDir="$4"

    cd $ObjDir
    echo "# Compiling llvm $Release-$RC $Flavor"
    echo "# ${MAKE} -j $NumJobs VERBOSE=1"
    ${MAKE} -j $NumJobs VERBOSE=1 \
        2>&1 | tee $LogDir/llvm.make-Phase$Phase-$Flavor.log

    echo "# Installing llvm $Release-$RC $Flavor"
    echo "# ${MAKE} install"
    ${MAKE} install \
        DESTDIR="${DestDir}" \
        2>&1 | tee $LogDir/llvm.install-Phase$Phase-$Flavor.log
    cd $BuildDir
}

function test_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"

    cd $ObjDir
    ${MAKE} -j $NumJobs -k check-all \
        2>&1 | tee $LogDir/llvm.check-Phase$Phase-$Flavor.log

    if [ "$use_autoconf" = "yes" ]; then
        # In the cmake build, unit tests are run as part of check-all.
        ${MAKE} -k unittests \
            2>&1 | tee $LogDir/llvm.unittests-Phase$Phase-$Flavor.log
    fi

    cd $BuildDir
}

# Clean RPATH. Libtool adds the build directory to the search path, which is
# not necessary --- and even harmful --- for the binary packages we release.
function clean_RPATH() {
  if [ `uname -s` = "Darwin" ]; then
    return
  fi
  local InstallPath="$1"
  for Candidate in `find $InstallPath/{bin,lib} -type f`; do
    if file $Candidate | grep ELF | egrep 'executable|shared object' > /dev/null 2>&1 ; then
      rpath=`objdump -x $Candidate | grep 'RPATH' | sed -e's/^ *RPATH *//'`
      if [ -n "$rpath" ]; then
        newrpath=`echo $rpath | sed -e's/.*\(\$ORIGIN[^:]*\).*/\1/'`
        chrpath -r $newrpath $Candidate 2>&1 > /dev/null 2>&1
      fi
    fi
  done
}

# Create a package of the release binaries.
function package_release() {
    cwd=`pwd`
    cd $BuildDir/Phase3/Release
    mv llvmCore-$Release-$RC.install $Package
    if [ "$use_gzip" = "yes" ]; then
      tar cfz $BuildDir/$Package.tar.gz $Package
    else
      tar cfJ $BuildDir/$Package.tar.xz $Package
    fi
    mv $Package llvmCore-$Release-$RC.install
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
            # case there are build paths in the debug info.
            if ! cmp --ignore-initial=16 <(sed -e 's,Phase2,Phase3,g' $p2) $p3 \
                    > /dev/null 2>&1 ; then
                echo "file `basename $p2` differs between phase 2 and phase 3"
            fi
        done
    fi
done
) 2>&1 | tee $LogDir/testing.$Release-$RC.log

package_release

set +e

# Woo hoo!
echo "### Testing Finished ###"
if [ "$use_gzip" = "yes" ]; then
  echo "### Package: $Package.tar.gz"
else
  echo "### Package: $Package.tar.xz"
fi
echo "### Logs: $LogDir"
exit 0
