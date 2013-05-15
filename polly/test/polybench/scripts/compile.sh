#!/bin/sh

if [ $# -ne 3 ]; then
    echo "Usage: compile.sh <compiler command> <input file> <output file>";
    exit 1;
fi;

COMPILER_COMMAND="$1";
INPUT_FILE="$2";
OUTPUT_FILE="$3";

$COMPILER_COMMAND -DPOLYBENCH_TIME -lm -I utilities utilities/instrument.c $INPUT_FILE -o $OUTPUT_FILE

exit 0;
