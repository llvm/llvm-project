#!/bin/bash

FILE=""
S_OPT=""
ORIG_ARGS="$@"
for arg in $@; do
  shift
  if [[ $arg == *".ll" ]]; then
      FILE="${arg}"
  fi
done

if [[ "$OSTYPE" == "darwin"* ]]; then
  # Mac
  TV_SHAREDLIB=tv.dylib
else
  # Linux, Cygwin/Msys, or Win32?
  TV_SHAREDLIB=tv.so
fi

TV_REPORT_DIR=""
TIMEOUT=""
TV_SMT_TO=""
TV_SMT_STATS=""
TV_REPORT_DIR=-tv-report-dir=alive2/build/logs
TIMEOUT=""
TV_SMT_TO="-tv-smt-to=100000"
TV_SMT_STATS=-tv-smt-stats

NPM_PLUGIN="-load-pass-plugin=alive2/build/tv/$TV_SHAREDLIB"

# Write input to temporary file so it can be passed to multiple opt calls, even
# if read from stdin. Run opt with original args, save output.
OUT=$(mktemp)
if [[ $FILE == "" ]]; then
    FILE="$(mktemp)"
    bin/opt > $FILE
    bin/opt $ORIG_ARGS $FILE > $OUT
else
  bin/opt $ORIG_ARGS > $OUT
fi

# Check if replacing all input functions with unreachable still verifies. If it does, the input has likely unconditional UB.
bin/opt -load=live2/build/tv/$TV_SHAREDLIB  $NPM_PLUGIN -tv-exit-on-error -passes="tv,to-unreachable,tv" -disable-output $FILE $TV_SMT_TO $TV_REPORT_DIR $TV_SMT_STATS 2> /dev/null
ret=$?

cat $OUT

if [ $ret -ne 0 ]; then
    exit 0
fi
exit 1
