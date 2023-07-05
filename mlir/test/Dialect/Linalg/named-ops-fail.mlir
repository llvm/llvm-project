// RUN: not mlir-opt -split-input-file -verify-diagnostics %s 2>&1 | FileCheck %s

func.func @add_type_cast(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf16>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op requires the same type for all operands and results
  linalg.add ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf16>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @add_broadcast(%arg0: memref<8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.add ins(%arg0, %arg1 : memref<8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

