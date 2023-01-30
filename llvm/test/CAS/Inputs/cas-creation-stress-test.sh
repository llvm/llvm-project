#!/bin/sh

# parameters are: <llvm-cas path> <CAS path> <file to ingest> <number of repeats>
LLVM_CAS_TOOL=$1
CAS_PATH=$2
INGEST_FILE=$3
NUM_REPEAT=$4

set -e

for c in $(seq 1 $NUM_REPEAT); do
  rm -rf $CAS_PATH

  pids=""
  for x in $(seq 1 10); do
    $LLVM_CAS_TOOL --ingest --data $INGEST_FILE --cas $CAS_PATH &
    pids="$pids $!"
  done

  for pid in $pids; do
    wait "$pid"
  done
done
