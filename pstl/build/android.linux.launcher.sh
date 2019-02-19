#!/bin/sh
#===-- android.linux.launcher.sh -----------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##
#
#

# Usage:
# android.linux.launcher.sh [-v] [-q] [-s] [-r <repeats>] [-u] [-l <library>] <executable> <arg1> <arg2> <argN>
#         where: -v enables verbose output
#         where: -q enables quiet mode
#         where: -s runs the test in stress mode (until non-zero exit code or ctrl-c pressed)
#         where: -r <repeats> specifies number of times to repeat execution
#         where: -u is ignored on Android
#         where: -l <library> specifies the library name to be assigned to LD_PRELOAD
#
# Libs and executable necessary for testing should be present in the current directory before running.
# ANDROID_SERIAL must be set to the connected Android target device name for file transfer and test runs.
# ANDROID_TEST_DIRECTORY may be set to the directory used for testing on the Android target device; otherwise,
#                        the default directory used is "/data/local/tmp/$(basename $PWD)".
# Note: Do not remove the redirections to '/dev/null' in the script, otherwise the nightly test system will fail.

do_cleanup() #
{ #
    adb pull $targetdir/events.txt events.txt > /dev/null 2>&1 #
    # Remove target directory on the device
    adb shell "rm -r ${targetdir}; mkdir -p ${targetdir}" > /dev/null 2>&1 #
} #
do_trap_cleanup() #
{ #
    do_cleanup #
    exit -1 #
} #
while getopts  "qvsr:ul:" flag #
do case $flag in #
    s )  # Stress testing mode
         echo Doing stress testing. Press Ctrl-C to terminate
         run_env='stressed() { while $*; do :; done; }; ' #
         run_prefix="stressed $run_prefix" ;; #
    r )  # Repeats test n times
         run_env="repeated() { for i in $(seq -s ' ' 1 $OPTARG) ; do echo \$i of $OPTARG:; \$*; done; }; " #
         run_prefix="repeated $run_prefix" ;; #
    l )  # Additional library
         ldpreload="$OPTARG " ;; #
    u )  # Stack limit
         ;; #
    q )  # Quiet mode, removes 'done' but prepends any other output by test name
         OUTPUT='2>&1 | sed -e "s/done//;/^[[:space:]]*$/d;s!^!$exename: !"' ;; #
    v )  # Verbose mode
         SUPPRESS='' #
         verbose=1 ;; #
esac done #
shift `expr $OPTIND - 1` #
[ -z "$OUTPUT" ] && OUTPUT='| sed -e "s/\\r$//"' #
[ $verbose ] || SUPPRESS='>/dev/null' #
# Collect the executable name
exename=$(basename $1) #
shift #

# Prepare the target directory on the device
currentdir=$(basename $PWD) #
targetdir=${ANDROID_TEST_DIRECTORY:-/data/local/tmp/$currentdir} #
do_cleanup #
trap do_trap_cleanup INT  # if someone hits control-c, cleanup the device

# Collect the list of files to transfer to the target device, starting with executable itself.
fnamelist="$exename" #
# Add the C++ standard library from the NDK, which is required for all tests on Android.
if [ ! -z "${LIB_STL_ANDROID}" ]; then #
    fnamelist="$fnamelist ${LIB_STL_ANDROID}" #
else #
    fnamelist="$fnamelist libc++_shared.so" #
fi #

# Find the TBB libraries and add them to the list.

OLD_SEP=$IFS
IFS=':'
for dir in $LD_LIBRARY_PATH; do #
    found="`ls $dir/lib*.so 2>/dev/null` "||: #
    fnamelist+="$fnamelist $found"
done #
IFS=$OLD_SEP

files="$(ls libtbb* 2> /dev/null)" #
[ -z "$files" ] || fnamelist="$fnamelist $files" #

# Add any libraries built for specific tests.
exeroot=${exename%\.*} #
files="$(ls ${exeroot}*.so ${exeroot}*.so.* 2> /dev/null)" #
[ -z "$files" ] || fnamelist="$fnamelist $files" #

# Transfer collected executable and library files to the target device.
transfers_ok=1 #
for fullname in $fnamelist; do { #
    if [ -r $fullname ]; then { #
        # Transfer the executable and libraries to top-level target directory
        if [ "$OS" = 'Windows_NT' ]; then #
            fullname=`cygpath -m "$fullname"` #
        fi #
        [ $verbose ] && echo -n "Pushing $fullname: " #
        eval "adb push $fullname ${targetdir}/$(basename $fullname) $SUPPRESS 2>&1" #
    }; else { #
        echo "Error: required file ${currentdir}/${fullname} for test $exename not available for transfer." #
        transfers_ok=0 #
    }; fi #
}; done #
if [ "${transfers_ok}" = "0" ]; then { #
    do_cleanup #
    exit -1 #
}; fi #
# Transfer input files used by example codes by scanning the executable argument list.
for fullname in "$@"; do { #
    if [ -r $fullname ]; then { #
        directory=$(dirname $fullname) #
        filename=$(basename $fullname) #
        # strip leading "." from fullname if present
        if [ "$directory" = "\." ]; then { #
            directory="" #
            fullname=$filename #
        }; fi #
        # Create the target directory to hold input file if necessary
        if [ ! -z $directory ]; then { #
            eval "adb shell 'mkdir $directory' $SUPPRESS 2>&1" #
        }; fi #
        # Transfer the input file to corresponding directory on target device
        [ $verbose ] && echo -n "Pushing $fullname: " #
        eval "adb push $fullname ${targetdir}/$fullname $SUPPRESS 2>&1" #
    }; fi #
}; done #

# Set LD_PRELOAD if necessary
[ -z "$ldpreload" ] || run_prefix="LD_PRELOAD='$ldpreload' $run_prefix" #
[ $verbose ] && echo Running $run_prefix ./$exename $* #
run_env="$run_env cd $targetdir; export LD_LIBRARY_PATH=." #
[ -z "$VIRTUAL_MACHINE" ] || run_env="$run_env; export VIRTUAL_MACHINE=$VIRTUAL_MACHINE" #
# The return_code file is the best way found to return the status of the test execution when using adb shell.
eval 'adb shell "$run_env; $run_prefix ./$exename $* || echo -n \$? >error_code"' "${OUTPUT}" #
# Capture the return code string and remove the trailing \r from the return_code file contents
err=`adb shell "cat $targetdir/error_code 2>/dev/null"` #
[ -z $err ] || echo $exename: exited with error $err #
do_cleanup #
# Return the exit code of the test.
exit $err #
