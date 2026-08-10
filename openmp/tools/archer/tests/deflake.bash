#!/usr/bin/env bash
# This script is used to deflake inherently flaky archer tests.
# It is invoked from lit tests as:
# %deflake mybinary
# which is then substituted by lit to:
# $(dirname %s)/deflake.bash mybinary
# The script runs the target program up to 10 times,
# until it fails (i.e. produces a race report).

for i in $(seq 1 10); do
    OUT=`$@ 2>&1`
    if [[ $? != 0 ]]; then
	echo "$OUT"
	exit 0
    fi
done
exit 1
