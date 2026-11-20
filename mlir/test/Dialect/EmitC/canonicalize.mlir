// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

func.func @no_fold_cast_to_f32(%arg0: i32) {
  // CHECK: emitc.cast
  %1 = emitc.cast %arg0: i32 to f32
  return
}

func.func @fold_cast_to_i64(%arg0: i32) {
  // CHECK-NOT: emitc.cast
  %1 = emitc.cast %arg0 pure : i32 to i64
  return
}

// CHECK-LABEL: func.func @erase_dead_if
// CHECK-NEXT:    return
func.func @erase_dead_if(%arg0: i1) {
  emitc.if %arg0 {
  }
  return
}
