#!/bin/bash -u

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <build dir> <file or dir path> [rejects dir] [checks]"
  echo " - <build dir> has to be a LLVM build directory (you should use CCACHE!)."
  echo " - <file or dir path> is the path that contains the .cpp files to update."
  echo " - [rejects dir] is a directory where rejected patch (build failure) will be stored."
  echo " - [checks] is an optional space-separated list of check to use instead of auto-detecting"
  echo " Also define the env var CLANG_TIDY the path to use for the clang-tidy binary (default to 'clang-tidy' in the PATH)"
  echo " Also define the env var TIMING_TIDY to 'time' to prefix clang-tidy execution with it"
  echo ""
  echo "This tool will execute clang-tidy on every .cpp file in the provided path and"
  echo "rerun the tests. On success, a commit is added to the repo for each individual"
  echo "pair <clang-tidy check, file>."
  exit 1
fi
BUILD_DIR=$1
SRCS=$2
REJECT_DIR=${3:-}
PRESET_CHECKS=${4:-}
SRC_DIR=$PWD
if [[ -v CLANG_TIDY ]] && [[ ! -z "$CLANG_TIDY" ]] ; then
  CLANG_TIDY=$(realpath $CLANG_TIDY)
  if [[ ! -f "$CLANG_TIDY" ]]; then
    echo "Invalid path '$CLANG_TIDY'"
    exit 1
  fi
else
  CLANG_TIDY=clang-tidy
fi
TIMING_TIDY=${TIMING_TIDY:-}
echo "Using: '$CLANG_TIDY"

if [[ ! -z "$REJECT_DIR" ]] && [[ ! -d "$REJECT_DIR" ]]; then
  echo "Expects 'rejects dir' to be a directory, got '$REJECT_DIR'"
  exit 1
fi

ensure_clean_build() {
  git reset --hard HEAD
  time ninja -C $BUILD_DIR check-mlir-build-only > ${REJECT_DIR}/ninja.clean.log 2>&1
  if [[ $? != 0 ]] ; then
    echo "-- Build failed on clean state, cleaning TableGen files and retry"
    # Reinitialize the TableGen generated file to have a clean state
    find $BUILD_DIR/tools/mlir/ | grep '\.inc' | while read file ; do rm $file ; done
    time ninja -C $BUILD_DIR check-mlir-build-only > ${REJECT_DIR}/ninja.clean.log 2>&1
    if [[ $? != 0 ]] ; then
      echo "check-mlir-build-only failed on clean state! (see ninja.clean.log)"
      git status
      exit 1
    fi
  fi
}

tmpfile=$(mktemp /tmp/mhlo-temp-checks.XXXXXX)
find $SRCS | grep ".cpp$" | sort | while read file ; do
  echo "================================"
  echo "======= Processing $file ======="
  date
  echo "================================"
  CHECKS=
  if [[ ! -z "$PRESET_CHECKS" ]]; then
    CHECKS="$PRESET_CHECKS"
  else
    CHECKS=$($CLANG_TIDY $file -p $BUILD_DIR --list-checks \
              | grep -v "Enabled checks:"  | grep -v "^$" \
              | while read check ; do echo -n "${check} " ; done;)
  fi
  echo "-----------------------------------"
  echo "-- Reset state before applying all checks on file $file"
  ensure_clean_build

  echo "-----------------------------------"
  echo "-- Apply all checks on file $file"
  echo "$TIMING_TIDY $CLANG_TIDY -p $BUILD_DIR $file -fix"
  $TIMING_TIDY $CLANG_TIDY -p $BUILD_DIR $file -fix \
    | grep "warning:.*\]$" | sed -r 's#.*\[(.*)]$#\1#' | sort -u > $tmpfile
  git clang-format -f
  if [[ $(git diff --stat) == '' ]]; then
    echo 'Nothing was applied, skip'
    continue
  fi
  echo "-----------------------------------"
  echo "-- Got some diff, run one check at a time now"
  cat $tmpfile | while read check ; do
    echo "-----------------------------------"
    echo "-- Reset state before applying check $check on file $file"
    ensure_clean_build

    echo "-----------------------------------"
    echo "-- Apply check $check on file $file"
    echo "$TIMING_TIDY $CLANG_TIDY -p $BUILD_DIR $file --checks="-*,$check" -fix"
    { $TIMING_TIDY $CLANG_TIDY -p $BUILD_DIR $file --checks="-*,$check" -fix ; } 2>&1
    git clang-format -f
    if [[ $(git diff --stat) == '' ]]; then
      echo 'Nothing was applied, skip'
      continue
    fi
    echo "-----------------------------------"
    echo "-- Test check $check on file $file"
    # Clang-tidy sometimes update files in the build directory, erase the .inc file generate by tablegen
    # to force them to be regenerated now.
    find $BUILD_DIR/tools/mlir/ | grep '\.inc' | while read file ; do rm $file ; done
    ninja -C $BUILD_DIR check-mlir > ${REJECT_DIR}/ninja.${check}.$(basename $file).log 2>&1
    if [[ $? != 0 ]] ; then
      echo "check-mlir failed! (see ninja.${check}.${file}.log)"
      [[ ! -z "$REJECT_DIR" ]] && git diff > "${REJECT_DIR}/${check}_$(basename ${file}).reject.diff"
      continue
    fi
    echo "-----------------------------------"
    echo "-- Success, commit changes for check $check on file $file"
    git clang-format -f

    git commit -a -m "Apply clang-tidy fixes for $check in $(basename $file) (NFC)"
  done
done
