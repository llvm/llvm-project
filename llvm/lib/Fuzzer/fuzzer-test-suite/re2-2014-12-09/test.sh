#!/bin/bash
set -x
SCRIPT_DIR=$(dirname $0)
EXECUTABLE_NAME_BASE=$(basename $SCRIPT_DIR)
CORPUS=CORPUS-$EXECUTABLE_NAME_BASE
JOBS=8
rm -rf $CORPUS
mkdir $CORPUS
[ -e $EXECUTABLE_NAME_BASE ] && ./$EXECUTABLE_NAME_BASE -exit_on_src_pos=re2/dfa.cc:474 -exit_on_src_pos=re2/dfa.cc:474  -runs=1000000 -jobs=$JOBS -workers=$JOBS $CORPUS
grep "INFO: found line matching 're2/dfa.cc:474', exiting." fuzz-0.log
