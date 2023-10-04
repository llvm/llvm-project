#!/bin/sh

# parameters are: <llvm-cas path> <CAS path> <file to ingest> <number of repeats>
LLVM_CAS_TOOL=$1
CAS_PATH=$2
INGEST_FILE=$3
NUM_REPEAT=$4

set -e

for c in $(seq 1 $NUM_REPEAT); do
  # Half the time, start from an existing CAS.
  if [[ $(($c % 2)) -eq 0 ]]; then
    rm -rf $CAS_PATH
  fi

  pids=""
  for x in $(seq 1 10); do
    $LLVM_CAS_TOOL --ingest $INGEST_FILE --cas $CAS_PATH &
    pids="$pids $!"
  done

  for pid in $pids; do
    wait "$pid"
  done

  cas_kb=$(du -ks $CAS_PATH | sed 's|\t.*$||')
  limit_in_kb=$((100 * 1024))
  # Check that the CAS is smaller than 100 MB, to ensure it was resized.
  if [[ $cas_kb -gt $limit_in_kb ]]; then
    echo "error: on-disk cas size ${cas_kb}kb was larger than ${limit_in_kb}kb" >&2
    exit 1
  fi
done
