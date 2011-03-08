#!/bin/bash
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

set -e                          # Exit if any command fails

Release=""
Release_no_dot=""
RC=""
do_checkout="yes"
do_ada="no"
do_objc="yes"
do_fortran="yes"
do_64bit="yes"
BuildDir="`pwd`"

function usage() {
    echo "usage: `basename $0` -release X.Y -rc NUM [OPTIONS]"
    echo ""
    echo " -release X.Y      The release number to test."
    echo " -rc NUM           The pre-release candidate number."
    echo " -j NUM            Number of compile jobs to run. [default: 3]"
    echo " -build-dir DIR    Directory to perform testing in. [default: pwd]"
    echo " -no-checkout      Don't checkout the sources from SVN."
    echo " -no-64bit         Don't test the 64-bit version. [default: yes]"
    echo " -enable-ada       Build Ada. [default: disable]"
    echo " -disable-objc     Disable ObjC build. [default: enable]"
    echo " -disable-fortran  Disable Fortran build. [default: enable]"
}

while [ $# -gt 0 ]; do
    case $1 in
        -release | --release )
            shift
            Release="$1"
            Release_no_dot="`echo $1 | sed -e 's,\.,,'`"
            ;;
        -rc | --rc | -RC | --RC )
            shift
            RC=$1
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
        -no-64bit | --no-64bit )
            do_64bit="no"
            ;;
        -enable-ada | --enable-ada )
            do_ada="yes"
            ;;
        -disable-objc | --disable-objc )
            do_objc="no"
            ;;
        -disable-fortran | --disable-fortran )
            echo "WARNING: Do you *really* need to disable Fortran?"
            sleep 5
            do_fortran="no"
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
    echo "No release number specified!"
    exit 1
fi
if [ -z "$RC" ]; then
    echo "No release candidate number specified!"
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

# Location of sources.
llvmCore_srcdir=$BuildDir/llvmCore-$Release-rc$RC.src
llvmgcc42_srcdir=$BuildDir/llvmgcc42-$Release-rc$RC.src

# SVN URLs for the sources.
Base_url="http://llvm.org/svn/llvm-project"
llvmCore_RC_url="$Base_url/llvm/tags/RELEASE_$Release_no_dot/rc$RC"
llvmgcc42_RC_url="$Base_url/llvm-gcc-4.2/tags/RELEASE_$Release_no_dot/rc$RC"
clang_RC_url="$Base_url/cfe/tags/RELEASE_$Release_no_dot/rc$RC"
test_suite_RC_url="$Base_url/test-suite/tags/RELEASE_$Release_no_dot/rc$RC"

# Make sure that the URLs are valid.
function check_valid_urls() {
    echo "# Validating SVN URLs"
    if ! svn ls $llvmCore_RC_url > /dev/null 2>&1 ; then
        echo "llvm $Release release candidate $RC doesn't exist!"
        exit 1
    fi
    if ! svn ls $llvmgcc42_RC_url > /dev/null 2>&1 ; then
        echo "llvm-gcc-4.2 $Release release candidate $RC doesn't exist!"
        exit 1
    fi
    if ! svn ls $clang_RC_url > /dev/null 2>&1 ; then
        echo "clang $Release release candidate $RC doesn't exist!"
        exit 1
    fi
    if ! svn ls $test_suite_RC_url > /dev/null 2>&1 ; then
        echo "test-suite $Release release candidate $RC doesn't exist!"
        exit 1
    fi
}

# Export sources to the the build directory.
function export_sources() {
    check_valid_urls

    echo "# Exporting llvm $Release-RC$RC sources"
    svn export -q $llvmCore_RC_url $llvmCore_srcdir
    echo "# Exporting llvm-gcc-4.2 $Release-rc$RC  sources"
    svn export -q $llvmgcc42_RC_url $llvmgcc42_srcdir
    echo "# Exporting clang $Release-rc$RC sources"
    svn export -q $clang_RC_url $llvmCore_srcdir/tools/clang
    echo "# Exporting llvm test suite $Release-rc$RC sources"
    svn export -q $test_suite_RC_url $llvmCore_srcdir/projects/llvm-test
}

function configure_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"
    InstallDir="$4"
    llvmgccDir="$5"

    case $Flavor in
        Release | Release-64 )
            Optimized="yes"
            Assertions="no"
            ;;
        Release+Asserts )
            Optimized="yes"
            Assertions="yes"
            ;;
        Debug )
            Optimized="no"
            Assertions="yes"
            ;;
        * )
            echo "# Invalid flavor $Flavor!"
            echo ""
            return
            ;;
    esac

    cd $ObjDir
    echo "# Configuring llvm $Release-rc$RC $Flavor"
    echo "# $llvmCore_srcdir/configure --prefix=$InstallDir \
        --enable-optimized=$Optimized \
        --enable-assertions=$Assertions \
        --with-llvmgccdir=$llvmgccDir"
    $llvmCore_srcdir/configure --prefix=$InstallDir \
        --enable-optimized=$Optimized \
        --enable-assertions=$Assertions \
        --with-llvmgccdir=$llvmgccDir
    cd -
}

function build_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"
    ExtraOpts=""

    CompilerFlags=""
    if [ "$Phase" = "2" ]; then
        CompilerFlags="CC=$llvmgccDir/bin/llvm-gcc CXX=$llvmgccDir/bin/llvm-g++"
    fi
    if [ "$Flavor" = "Release-64" ]; then
        ExtraOpts="EXTRA_OPTIONS=-m64"
    fi

    cd $ObjDir
    echo "# Compiling llvm $Release-rc$RC $Flavor"
    echo "# make -j $NumJobs VERBOSE=1 $ExtraOpts"
    make -j $NumJobs VERBOSE=1 $ExtraOpts $CompilerFlags

    echo "# Installing llvm $Release-rc$RC $Flavor"
    echo "# make install"
    make install
    cd -
}

function test_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"

    cd $ObjDir
    make check
    make -C tools/clang test
    make unittests
    cd -
}

function configure_llvm_gcc() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"
    InstallDir="$4"
    llvmObjDir="$5"

    languages="c,c++"
    if [ "$do_objc" = "yes" ]; then
        languages="$languages,objc,obj-c++"
    fi
    if [ "$do_fortran" = "yes" ]; then
        languages="$languages,fortran"
    fi
    if [ "$do_ada" = "yes" ]; then
        languages="$languages,ada"
    fi

    cd $ObjDir
    echo "# Configuring llvm-gcc $Release-rc$RC $Flavor"
    echo "# $llvmgcc42_srcdir/configure --prefix=$InstallDir \
        --program-prefix=llvm- --enable-llvm=$llvmObjDir \
        --enable-languages=$languages"
    $llvmgcc42_srcdir/configure --prefix=$InstallDir \
        --program-prefix=llvm- --enable-llvm=$llvmObjDir \
        --enable-languages=$languages
    cd -
}

function build_llvm_gcc() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"
    llvmgccDir="$4"

    CompilerFlags=""
    if [ "$Phase" = "2" ]; then
        CompilerFlags="CC=$llvmgccDir/bin/llvm-gcc CXX=$llvmgccDir/bin/llvm-g++"
    fi

    cd $ObjDir
    echo "# Compiling llvm-gcc $Release-rc$RC $Flavor"
    echo "# make -j $NumJobs bootstrap LLVM_VERSION_INFO=$Release"
    make -j $NumJobs bootstrap LLVM_VERSION_INFO=$Release $CompilerFlags

    echo "# Installing llvm-gcc $Release-rc$RC $Flavor"
    echo "# make install"
    make install
    cd -
}

if [ "$do_checkout" = "yes" ]; then
    export_sources
fi

Flavors="Debug Release Release+Asserts"
if [ "$do_64bit" = "yes" ]; then
    Flavors="$Flavors Release-64"
fi

for Flavor in $Flavors ; do
    echo ""
    echo ""
    echo "********************************************************************************"
    echo "  Release:     $Release-rc$RC"
    echo "  Build:       $Flavor"
    echo "  System Info: "
    echo "    `uname -a`"
    echo "********************************************************************************"
    echo ""

    llvmCore_phase1_objdir=$BuildDir/Phase1/$Flavor/llvmCore-$Release-rc$RC.obj
    llvmCore_phase1_installdir=$BuildDir/Phase1/$Flavor/llvmCore-$Release-rc$RC.install

    llvmCore_phase2_objdir=$BuildDir/Phase2/$Flavor/llvmCore-$Release-rc$RC.obj
    llvmCore_phase2_installdir=$BuildDir/Phase2/$Flavor/llvmCore-$Release-rc$RC.install

    rm -rf $llvmCore_phase1_objdir
    rm -rf $llvmCore_phase1_installdir
    rm -rf $llvmCore_phase2_objdir
    rm -rf $llvmCore_phase2_installdir

    mkdir -p $llvmCore_phase1_objdir
    mkdir -p $llvmCore_phase1_installdir
    mkdir -p $llvmCore_phase2_objdir
    mkdir -p $llvmCore_phase2_installdir

    llvmgcc42_phase1_objdir=$BuildDir/Phase1/$Flavor/llvmgcc42-$Release-rc$RC.obj
    llvmgcc42_phase1_installdir=$BuildDir/Phase1/$Flavor/llvmgcc42-$Release-rc$RC.install

    llvmgcc42_phase2_objdir=$BuildDir/Phase2/$Flavor/llvmgcc42-$Release-rc$RC.obj
    llvmgcc42_phase2_installdir=$BuildDir/Phase2/$Flavor/llvmgcc42-$Release-rc$RC.install

    rm -rf $llvmgcc42_phase1_objdir
    rm -rf $llvmgcc42_phase1_installdir
    rm -rf $llvmgcc42_phase2_objdir
    rm -rf $llvmgcc42_phase2_installdir

    mkdir -p $llvmgcc42_phase1_objdir
    mkdir -p $llvmgcc42_phase1_installdir
    mkdir -p $llvmgcc42_phase2_objdir
    mkdir -p $llvmgcc42_phase2_installdir

    ############################################################################
    # Phase 1: Build llvmCore and llvmgcc42
    echo "# Phase 1: Building llvmCore"
    configure_llvmCore 1 $Flavor \
        $llvmCore_phase1_objdir $llvmCore_phase1_installdir \
        $llvmgcc42_phase1_installdir
    build_llvmCore 1 $Flavor \
        $llvmCore_phase1_objdir

    echo "# Phase 1: Building llvmgcc42"
    configure_llvm_gcc 1 $Flavor \
        $llvmgcc42_phase1_objdir $llvmgcc42_phase1_installdir \
        $llvmCore_phase1_objdir
    build_llvm_gcc 1 $Flavor \
        $llvmgcc42_phase1_objdir $llvmgcc42_phase1_installdir

    ############################################################################
    # Phase 2: Build llvmCore with newly built llvmgcc42 from phase 1.
    echo "# Phase 2: Building llvmCore"
    configure_llvmCore 2 $Flavor \
        $llvmCore_phase2_objdir $llvmCore_phase2_installdir \
        $llvmgcc42_phase1_installdir
    build_llvmCore 2 $Flavor \
        $llvmCore_phase2_objdir

    echo "# Phase 2: Building llvmgcc42"
    configure_llvm_gcc 2 $Flavor \
        $llvmgcc42_phase2_objdir $llvmgcc42_phase2_installdir \
        $llvmCore_phase2_objdir
    build_llvm_gcc 2 $Flavor \
        $llvmgcc42_phase2_objdir $llvmgcc42_phase1_installdir

    echo "# Testing - built with llvmgcc42"
    test_llvmCore 2 $Flavor $llvmCore_phase2_objdir
done

# Woo hoo!
echo "### Testing Finished ###"
exit 0
