#!/usr/bin/env bash
# For each testcase:  C --clang--> .ll --opt local-opt--> .opt.ll,
# then FileCheck compares .opt.ll with the CHECK lines in the testcase.
cd "$(dirname "$0")"
LLVM=${LLVM:-../../build/bin}
PLUGIN=${PLUGIN:-../build/LocalOptimization.so}
OUT=../build/testcases
mkdir -p $OUT

for src in *.c; do
  name=${src%.c}
  # -disable-O0-optnone: otherwise opt skips every -O0 function.
  $LLVM/clang -S -emit-llvm -O0 -Xclang -disable-O0-optnone \
      -fno-discard-value-names $src -o $OUT/$name.ll &&
  $LLVM/opt -load-pass-plugin=$PLUGIN -passes=local-opt -S \
      $OUT/$name.ll -o $OUT/$name.opt.ll &&
  $LLVM/FileCheck --input-file=$OUT/$name.opt.ll $src &&
  echo "PASS $name" || echo "FAIL $name"
done
