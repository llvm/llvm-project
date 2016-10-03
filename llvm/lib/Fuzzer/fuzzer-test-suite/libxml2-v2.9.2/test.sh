#!/bin/bash
set -x
. $(dirname $0)/../common.sh

get_git_revision https://github.com/mcarpenter/afl be3e88d639da5350603f6c0fee06970128504342 afl
rm -rf $CORPUS
mkdir $CORPUS
[ -e $EXECUTABLE_NAME_BASE ] && ./$EXECUTABLE_NAME_BASE -artifact_prefix=$CORPUS/ -jobs=$JOBS -dict=afl/dictionaries/xml.dict -workers=$JOBS $CORPUS
grep "AddressSanitizer: heap-buffer-overflow" fuzz-0.log
