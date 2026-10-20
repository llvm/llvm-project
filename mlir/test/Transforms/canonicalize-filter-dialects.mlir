// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize{filter-dialects=arith}))' | FileCheck %s --check-prefix=ARITH
// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize{filter-dialects=func}))' | FileCheck %s --check-prefix=FUNC
// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s --check-prefix=ALL
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{filter-dialects=does_not_exist}))' 2>&1 | FileCheck %s --check-prefix=ERR

// The `SubIRHSAddConstant` arith pattern rewrites `subi(addi(x, c0), c1)` into
// `addi(x, c0 - c1)`. The pattern only fires when arith canonicalizations are
// loaded.

// ARITH-LABEL: func @pattern_test
// ARITH-NOT:     arith.subi
// ARITH:         arith.addi %{{.*}}, %[[C:.*]]

// FUNC-LABEL: func @pattern_test
// FUNC:         arith.addi
// FUNC:         arith.subi

// ALL-LABEL: func @pattern_test
// ALL-NOT:     arith.subi
// ALL:         arith.addi %{{.*}}, %[[C:.*]]

// ERR: can't load dialect 'does_not_exist': missing registration?
func.func @pattern_test(%a: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %add = arith.addi %a, %c1 : i32
  %sub = arith.subi %add, %c2 : i32
  return %sub : i32
}
