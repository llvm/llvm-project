#!/bin/bash
. $(dirname $0)/../common.sh
set -x
[ -e $EXECUTABLE_NAME_BASE ] && ./$EXECUTABLE_NAME_BASE -max_total_time=10 2>&1 | tee log
grep -Pzo "(?s)ERROR: AddressSanitizer: heap-buffer-overflow.*WRITE of size 1.*ares_create_query.*is located 0 bytes to the right of" log
