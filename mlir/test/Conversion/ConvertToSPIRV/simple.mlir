// RUN: mlir-opt -test-convert-to-spirv="run-signature-conversion=false run-vector-unrolling=false" -split-input-file %s | FileCheck %s

// CHECK-LABEL: @return_scalar
// CHECK-SAME: %[[ARG0:.*]]: i32
// CHECK: spirv.ReturnValue %[[ARG0]]
func.func @return_scalar(%arg0 : i32) -> i32 {
  return %arg0 : i32
}

// CHECK-LABEL: @return_vector
// CHECK-SAME: %[[ARG0:.*]]: vector<4xi32>
// CHECK: spirv.ReturnValue %[[ARG0]]
func.func @return_vector(%arg0 : vector<4xi32>) -> vector<4xi32> {
  return %arg0 : vector<4xi32>
}

// CHECK-LABEL: @cond_br
// CHECK-SAME: %[[ARG0:.*]]: i1, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32
// CHECK:       spirv.BranchConditional %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK-NEXT:  ^[[BB1]]:
// CHECK-NEXT:    spirv.Branch ^[[BB3:.*]](%[[ARG1]] : i32)
// CHECK-NEXT:  ^[[BB2]]:
// CHECK-NEXT:    spirv.Branch ^[[BB3]](%[[ARG2]] : i32)
// CHECK-NEXT:  ^[[BB3]]
// CHECK:         spirv.ReturnValue
func.func @cond_br(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : i32)
^bb2:
  cf.br ^bb3(%arg2 : i32)
^bb3(%r: i32):
  return %r : i32
}
