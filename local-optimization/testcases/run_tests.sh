#!/usr/bin/env bash
# Runs every testcase (or the ones given as arguments):
#
#   testcase.c --clang -O0--> testcase.ll --opt local-opt--> checked with FileCheck
#
# The expectations are written at the bottom of each testcase:
#   TEST-OPTS: ...   the optimizations the ONLY checks are run with
#   ONLY...:         expected output of  -passes='local-opt<TEST-OPTS>'
#   ALL...:          expected output of  -passes=local-opt   (all five)
#
# Usage:  ./run_tests.sh [testcase.c ...]
#
# Paths can be overridden with these environment variables
# ("cmake --build build --target check" sets them automatically):
#   LLVM_BIN  directory containing clang, opt and FileCheck
#   PLUGIN    LocalOptimization.so
#   OUT_DIR   where the generated .ll files are written
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
LLVM_BIN=${LLVM_BIN:-$HERE/../../build/bin}
PLUGIN=${PLUGIN:-$HERE/../build/LocalOptimization.so}
OUT_DIR=${OUT_DIR:-$HERE/../build/testcases}
mkdir -p "$OUT_DIR"

[[ $# -eq 0 ]] && set -- "$HERE"/*.c

# $1 = pass pipeline, $2 = output file, $3 = FileCheck prefix
optimize_and_check() {
  "$LLVM_BIN/opt" -load-pass-plugin="$PLUGIN" -passes="$1" -S "$ll" -o "$2" &&
    "$LLVM_BIN/FileCheck" --check-prefix="$3" --input-file="$2" "$src"
}

passed=0
failed=0
for src in "$@"; do
  name=$(basename "$src" .c)
  ll=$OUT_DIR/$name.ll
  opts=$(sed -n 's|^// TEST-OPTS: *||p' "$src")

  # At -O0 clang marks every function "optnone", and opt then skips
  # optimization passes on it; -disable-O0-optnone prevents that.
  ok=1
  "$LLVM_BIN/clang" -S -emit-llvm -O0 -Xclang -disable-O0-optnone \
    -fno-discard-value-names "$src" -o "$ll" || ok=0
  if ((ok)); then
    optimize_and_check "local-opt<$opts>" "$OUT_DIR/$name.only.ll" ONLY || ok=0
    optimize_and_check "local-opt" "$OUT_DIR/$name.all.ll" ALL || ok=0
  fi

  if ((ok)); then
    printf 'PASS  %-22s local-opt<%s>, local-opt\n' "$name" "$opts"
    passed=$((passed + 1))
  else
    printf 'FAIL  %s\n' "$name"
    failed=$((failed + 1))
  fi
done

echo
echo "$passed passed, $failed failed.  Generated IR: $OUT_DIR"
((failed == 0))
