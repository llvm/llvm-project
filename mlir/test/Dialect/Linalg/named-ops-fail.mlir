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

// -----

func.func @sub_type_cast(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf16>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op requires the same type for all operands and results
  linalg.sub ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf16>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @sub_broadcast(%arg0: memref<8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.sub ins(%arg0, %arg1 : memref<8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @mul_type_cast(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf16>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op requires the same type for all operands and results
  linalg.mul ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf16>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @mul_broadcast(%arg0: memref<8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.mul ins(%arg0, %arg1 : memref<8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @div_type_cast(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf16>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op requires the same type for all operands and results
  linalg.div ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf16>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @div_broadcast(%arg0: memref<8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.div ins(%arg0, %arg1 : memref<8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @divu_type_cast(%arg0: memref<4x8x16xi32>, %arg1: memref<4x8x16xi16>, %arg2: memref<4x8x16xi32>) {
  // CHECK: op requires the same type for all operands and results
  linalg.div_unsigned ins(%arg0, %arg1 : memref<4x8x16xi32>, memref<4x8x16xi16>) outs(%arg2: memref<4x8x16xi32>)
  return
}

// -----

func.func @divu_broadcast(%arg0: memref<8x16xi32>, %arg1: memref<4x8x16xi32>, %arg2: memref<4x8x16xi32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.div_unsigned ins(%arg0, %arg1 : memref<8x16xi32>, memref<4x8x16xi32>) outs(%arg2: memref<4x8x16xi32>)
  return
}
