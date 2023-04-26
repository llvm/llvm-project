#!/bin/bash

ARRAY_SIZES=( 1 100 1024 2048 4096 8192 10000 81920 1000000 4194304 23445657 41943040 100000000 177777777 )

exit_code=0
for size in "${ARRAY_SIZES[@]}"; do
  ARRAY_SIZE=$size ./test_xteams.sh
  exit_code=$(( $? + $exit_code ))
  echo "cumulative exit_code: " $exit_code
done

if [ $exit_code != 0 ] ; then 
  echo "XXXXXXXX ERRORS OCCURRED XXXXXXXX"
else
  echo "<<<<<<<< ALL PASSED >>>>>>>>"
fi