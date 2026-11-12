// RUN: not mlir-opt -test-extract-fixed-outer-loops %s 2>&1 | FileCheck %s

func.func @no_crash(%arg0: memref<?x?xf32>) {
  // CHECK: error: missing `test-outer-loop-sizes` pass-option for outer loop sizes
  return
}
