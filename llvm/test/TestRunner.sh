#!/bin/sh
#
#  TestRunner.sh - This script is used to run the deja-gnu tests exactly like
#  deja-gnu does, by executing the Tcl script specified in the test case's 
#  RUN: lines. This is made possible by a simple make target supported by the
#  test/Makefile. All this script does is invoke that make target. 
#
#  Usage:
#     TestRunner.sh {script_names}
#
#     This script is typically used by cd'ing to a test directory and then
#     running TestRunner.sh with a list of test file names you want to run.
#
TESTPATH=`pwd`
SUBDIR=""
if test `dirname $1` = "." ; then
  while test `basename $TESTPATH` != "test" -a ! -z "$TESTPATH" ; do
    tmp=`basename $TESTPATH`
    SUBDIR="$tmp/$SUBDIR"
    TESTPATH=`dirname $TESTPATH`
  done
fi

for TESTFILE in "$@" ; do 
  if test `dirname $TESTFILE` = . ; then
    if test -d "$TESTPATH" ; then
      cd $TESTPATH
      make check-one TESTONE="$SUBDIR$TESTFILE"
      cd $PWD
    else
      echo "Can't find llvm/test directory in " `pwd`
    fi
  else
    make check-one TESTONE=$TESTFILE
  fi
done
