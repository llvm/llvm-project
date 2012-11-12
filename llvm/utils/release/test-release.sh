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

projects="llvm cfe dragonegg compiler-rt test-suite"

# Base SVN URL for the sources.
Base_url="http://llvm.org/svn/llvm-project"

Release=""
Release_no_dot=""
RC=""
do_checkout="yes"
do_ada="no"
do_clang="yes"
do_dragonegg="no"
do_fortran="no"
do_objc="yes"
do_64bit="yes"
do_debug="no"
do_asserts="no"
do_compare="yes"
BuildDir="`pwd`"

function usage() {
    echo "usage: `basename $0` -release X.Y -rc NUM [OPTIONS]"
    echo ""
    echo " -release X.Y      The release number to test."
    echo " -rc NUM           The pre-release candidate number."
    echo " -final            The final release candidate."
    echo " -j NUM            Number of compile jobs to run. [default: 3]"
    echo " -build-dir DIR    Directory to perform testing in. [default: pwd]"
    echo " -no-checkout      Don't checkout the sources from SVN."
    echo " -no-64bit         Don't test the 64-bit version. [default: yes]"
    echo " -enable-ada       Build Ada. [default: disable]"
    echo " -disable-clang    Do not test clang. [default: enable]"
    echo " -enable-dragonegg Test dragonegg. [default: disable]"
    echo " -enable-fortran   Enable Fortran build. [default: disable]"
    echo " -disable-objc     Disable ObjC build. [default: enable]"
    echo " -test-debug       Test the debug build. [default: no]"
    echo " -test-asserts     Test with asserts on. [default: no]"
    echo " -no-compare-files Don't test that phase 2 and 3 files are identical."
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
            RC="rc$1"
            ;;
        -final | --final )
            RC=final
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
        -disable-clang | --disable-clang )
            do_clang="no"
            ;;
        -enable-dragonegg | --enable-dragonegg )
            do_dragonegg="yes"
            ;;
        -enable-fortran | --enable-fortran )
            do_fortran="yes"
            ;;
        -disable-objc | --disable-objc )
            do_objc="no"
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

# Find compilers.
if [ "$do_dragonegg" = "yes" ]; then
    gcc_compiler="$GCC"
    if [ -z "$gcc_compiler" ]; then
        gcc_compiler="`which gcc`"
        if [ -z "$gcc_compiler" ]; then
            echo "error: cannot find gcc to use with dragonegg"
            exit 1
        fi
    fi

    gxx_compiler="$GXX"
    if [ -z "$gxx_compiler" ]; then
        gxx_compiler="`which g++`"
        if [ -z "$gxx_compiler" ]; then
            echo "error: cannot find g++ to use with dragonegg"
            exit 1
        fi
    fi
fi


# Make sure that the URLs are valid.
function check_valid_urls() {
    for proj in $projects ; do
        echo "# Validating $proj SVN URL"

        if ! svn ls $Base_url/$proj/tags/RELEASE_$Release_no_dot/$RC > /dev/null 2>&1 ; then
            echo "llvm $Release release candidate $RC doesn't exist!"
            exit 1
        fi
    done
}

# Export sources to the build directory.
function export_sources() {
    check_valid_urls

    for proj in $projects ; do
        echo "# Exporting $proj $Release-RC$RC sources"
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
    cd $BuildDir/llvm.src/projects
    if [ ! -h llvm-test ]; then
        ln -s ../../test-suite.src llvm-test
    fi
    if [ ! -h compiler-rt ]; then
        ln -s ../../compiler-rt.src compiler-rt
    fi
    cd $BuildDir
}

function configure_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"
    InstallDir="$4"

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
            echo "# Invalid flavor '$Flavor'"
            echo ""
            return
            ;;
    esac

    echo "# Using C compiler: $c_compiler"
    echo "# Using C++ compiler: $cxx_compiler"

    cd $ObjDir
    echo "# Configuring llvm $Release-$RC $Flavor"
    echo "# $BuildDir/llvm.src/configure --prefix=$InstallDir \
        --enable-optimized=$Optimized \
        --enable-assertions=$Assertions"
    env CC="$c_compiler" CXX="$cxx_compiler" \
    $BuildDir/llvm.src/configure --prefix=$InstallDir \
        --enable-optimized=$Optimized \
        --enable-assertions=$Assertions \
        --disable-timestamps \
        2>&1 | tee $LogDir/llvm.configure-Phase$Phase-$Flavor.log
    cd $BuildDir
}

function build_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"
    ExtraOpts=""

    if [ "$Flavor" = "Release-64" ]; then
        ExtraOpts="EXTRA_OPTIONS=-m64"
    fi

    cd $ObjDir
    echo "# Compiling llvm $Release-$RC $Flavor"
    echo "# ${MAKE} -j $NumJobs VERBOSE=1 $ExtraOpts"
    ${MAKE} -j $NumJobs VERBOSE=1 $ExtraOpts \
        2>&1 | tee $LogDir/llvm.make-Phase$Phase-$Flavor.log

    echo "# Installing llvm $Release-$RC $Flavor"
    echo "# ${MAKE} install"
    ${MAKE} install \
        2>&1 | tee $LogDir/llvm.install-Phase$Phase-$Flavor.log
    cd $BuildDir
}

function build_dragonegg() {
    Phase="$1"
    Flavor="$2"
    LLVMInstallDir="$3"
    DragonEggObjDir="$4"
    LLVM_CONFIG=$LLVMInstallDir/bin/llvm-config
    TOP_DIR=$BuildDir/dragonegg.src

    echo "# Targeted compiler: $gcc_compiler"

    cd $DragonEggObjDir
    echo "# Compiling phase $Phase dragonegg $Release-$RC $Flavor"
    echo -n "# CXX=$cxx_compiler TOP_DIR=$TOP_DIR GCC=$gcc_compiler "
    echo -n "LLVM_CONFIG=$LLVM_CONFIG ${MAKE} -f $TOP_DIR/Makefile "
    echo "-j $NumJobs VERBOSE=1"
    CXX="$cxx_compiler" TOP_DIR="$TOP_DIR" GCC="$gcc_compiler" \
    LLVM_CONFIG="$LLVM_CONFIG" ${MAKE} -f $TOP_DIR/Makefile \
        -j $NumJobs VERBOSE=1 \
            2>&1 | tee $LogDir/dragonegg-Phase$Phase-$Flavor.log
    cd $BuildDir
}

function test_llvmCore() {
    Phase="$1"
    Flavor="$2"
    ObjDir="$3"

    cd $ObjDir
    ${MAKE} -k check-all \
        2>&1 | tee $LogDir/llvm.check-Phase$Phase-$Flavor.log
    ${MAKE} -k unittests \
        2>&1 | tee $LogDir/llvm.unittests-Phase$Phase-$Flavor.log
    cd $BuildDir
}

set -e                          # Exit if any command fails

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
if [ "$do_64bit" = "yes" ]; then
    Flavors="$Flavors Release-64"
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
    llvmCore_phase1_installdir=$BuildDir/Phase1/$Flavor/llvmCore-$Release-$RC.install
    dragonegg_phase1_objdir=$BuildDir/Phase1/$Flavor/DragonEgg-$Release-$RC.obj

    llvmCore_phase2_objdir=$BuildDir/Phase2/$Flavor/llvmCore-$Release-$RC.obj
    llvmCore_phase2_installdir=$BuildDir/Phase2/$Flavor/llvmCore-$Release-$RC.install
    llvmCore_de_phase2_objdir=$BuildDir/Phase2/$Flavor/llvmCore-DragonEgg-$Release-$RC.obj
    llvmCore_de_phase2_installdir=$BuildDir/Phase2/$Flavor/llvmCore-DragonEgg-$Release-$RC.install
    dragonegg_phase2_objdir=$BuildDir/Phase2/$Flavor/DragonEgg-$Release-$RC.obj

    llvmCore_phase3_objdir=$BuildDir/Phase3/$Flavor/llvmCore-$Release-$RC.obj
    llvmCore_phase3_installdir=$BuildDir/Phase3/$Flavor/llvmCore-$Release-$RC.install
    llvmCore_de_phase3_objdir=$BuildDir/Phase3/$Flavor/llvmCore-DragonEgg-$Release-$RC.obj
    llvmCore_de_phase3_installdir=$BuildDir/Phase3/$Flavor/llvmCore-DragonEgg-$Release-$RC.install
    dragonegg_phase3_objdir=$BuildDir/Phase3/$Flavor/DragonEgg-$Release-$RC.obj

    rm -rf $llvmCore_phase1_objdir
    rm -rf $llvmCore_phase1_installdir
    rm -rf $dragonegg_phase1_objdir

    rm -rf $llvmCore_phase2_objdir
    rm -rf $llvmCore_phase2_installdir
    rm -rf $llvmCore_de_phase2_objdir
    rm -rf $llvmCore_de_phase2_installdir
    rm -rf $dragonegg_phase2_objdir

    rm -rf $llvmCore_phase3_objdir
    rm -rf $llvmCore_phase3_installdir
    rm -rf $llvmCore_de_phase3_objdir
    rm -rf $llvmCore_de_phase3_installdir
    rm -rf $dragonegg_phase3_objdir

    mkdir -p $llvmCore_phase1_objdir
    mkdir -p $llvmCore_phase1_installdir
    mkdir -p $dragonegg_phase1_objdir

    mkdir -p $llvmCore_phase2_objdir
    mkdir -p $llvmCore_phase2_installdir
    mkdir -p $llvmCore_de_phase2_objdir
    mkdir -p $llvmCore_de_phase2_installdir
    mkdir -p $dragonegg_phase2_objdir

    mkdir -p $llvmCore_phase3_objdir
    mkdir -p $llvmCore_phase3_installdir
    mkdir -p $llvmCore_de_phase3_objdir
    mkdir -p $llvmCore_de_phase3_installdir
    mkdir -p $dragonegg_phase3_objdir

    ############################################################################
    # Phase 1: Build llvmCore and clang
    echo "# Phase 1: Building llvmCore"
    configure_llvmCore 1 $Flavor \
        $llvmCore_phase1_objdir $llvmCore_phase1_installdir
    build_llvmCore 1 $Flavor \
        $llvmCore_phase1_objdir

    # Test clang
    if [ "$do_clang" = "yes" ]; then
        ########################################################################
        # Phase 2: Build llvmCore with newly built clang from phase 1.
        c_compiler=$llvmCore_phase1_installdir/bin/clang
        cxx_compiler=$llvmCore_phase1_installdir/bin/clang++
        echo "# Phase 2: Building llvmCore"
        configure_llvmCore 2 $Flavor \
            $llvmCore_phase2_objdir $llvmCore_phase2_installdir
        build_llvmCore 2 $Flavor \
            $llvmCore_phase2_objdir

        ########################################################################
        # Phase 3: Build llvmCore with newly built clang from phase 2.
        c_compiler=$llvmCore_phase2_installdir/bin/clang
        cxx_compiler=$llvmCore_phase2_installdir/bin/clang++
        echo "# Phase 3: Building llvmCore"
        configure_llvmCore 3 $Flavor \
            $llvmCore_phase3_objdir $llvmCore_phase3_installdir
        build_llvmCore 3 $Flavor \
            $llvmCore_phase3_objdir

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
            for o in `find $llvmCore_phase2_objdir -name '*.o'` ; do
                p3=`echo $o | sed -e 's,Phase2,Phase3,'`
                if ! cmp --ignore-initial=16 $o $p3 > /dev/null 2>&1 ; then
                    echo "file `basename $o` differs between phase 2 and phase 3"
                fi
            done
        fi
    fi

    # Test dragonegg
    if [ "$do_dragonegg" = "yes" ]; then
        # Build dragonegg using the targeted gcc.  This isn't necessary, but
        # helps avoid using broken versions of gcc (which are legion), tests
        # that the targeted gcc is basically sane and is consistent with the
        # later phases in which the targeted gcc + dragonegg are used.
        c_compiler="$gcc_compiler"
        cxx_compiler="$gxx_compiler"
        build_dragonegg 1 $Flavor $llvmCore_phase1_installdir $dragonegg_phase1_objdir

        ########################################################################
        # Phase 2: Build llvmCore with newly built dragonegg from phase 1.
        c_compiler="$gcc_compiler -fplugin=$dragonegg_phase1_objdir/dragonegg.so"
        cxx_compiler="$gxx_compiler -fplugin=$dragonegg_phase1_objdir/dragonegg.so"
        echo "# Phase 2: Building llvmCore with dragonegg"
        configure_llvmCore 2 $Flavor \
            $llvmCore_de_phase2_objdir $llvmCore_de_phase2_installdir
        build_llvmCore 2 $Flavor \
            $llvmCore_de_phase2_objdir
        build_dragonegg 2 $Flavor $llvmCore_de_phase2_installdir $dragonegg_phase2_objdir

        ########################################################################
        # Phase 3: Build llvmCore with newly built clang from phase 2.
        c_compiler="$gcc_compiler -fplugin=$dragonegg_phase2_objdir/dragonegg.so"
        cxx_compiler="$gxx_compiler -fplugin=$dragonegg_phase2_objdir/dragonegg.so"
        echo "# Phase 3: Building llvmCore with dragonegg"
        configure_llvmCore 3 $Flavor \
            $llvmCore_de_phase3_objdir $llvmCore_de_phase3_installdir
        build_llvmCore 3 $Flavor \
            $llvmCore_de_phase3_objdir
        build_dragonegg 3 $Flavor $llvmCore_de_phase3_installdir $dragonegg_phase3_objdir

        ########################################################################
        # Testing: Test phase 3
        c_compiler="$gcc_compiler -fplugin=$dragonegg_phase3_objdir/dragonegg.so"
        cxx_compiler="$gxx_compiler -fplugin=$dragonegg_phase3_objdir/dragonegg.so"
        echo "# Testing - built with dragonegg"
        test_llvmCore 3 $Flavor $llvmCore_de_phase3_objdir

        ########################################################################
        # Compare .o files between Phase2 and Phase3 and report which ones differ.
        echo
        echo "# Comparing Phase 2 and Phase 3 files"
        for o in `find $llvmCore_de_phase2_objdir -name '*.o'` \
          `find $dragonegg_phase2_objdir -name '*.o'` ; do
            p3=`echo $o | sed -e 's,Phase2,Phase3,'`
            if ! cmp --ignore-initial=16 $o $p3 > /dev/null 2>&1 ; then
                echo "file `basename $o` differs between dragonegg phase 2 and phase 3"
            fi
        done
    fi

    # Otherwise just test the core.
    if [ "$do_clang" != "yes" -a "$do_dragonegg" != "yes" ]; then
        echo "# Testing - built with system compiler"
        test_llvmCore 1 $Flavor $llvmCore_phase1_objdir
    fi
done
) 2>&1 | tee $LogDir/testing.$Release-$RC.log

set +e

# Woo hoo!
echo "### Testing Finished ###"
echo "### Logs: $LogDir"
exit 0
