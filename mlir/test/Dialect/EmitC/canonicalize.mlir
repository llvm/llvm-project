// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

// While there is no dedicated folder for CastOp, it is Pure and hence should
// be "folded away".
func.func @cast(%arg0: i32) {
  // CHECK-NOT: emitc.cast
  %1 = emitc.cast %arg0: i32 to f32
  return
}
