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

// -----

func.func @exp_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.exp ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @exp_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.exp ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @log_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.log ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @log_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.log ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @abs_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.abs ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @abs_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.abs ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @ceil_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.ceil ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @ceil_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.ceil ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @floor_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.floor ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @floor_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.floor ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @negf_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.negf ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @negf_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.negf ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @reciprocal_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.reciprocal ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @reciprocal_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.reciprocal ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @round_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.round ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @round_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.round ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @sqrt_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.sqrt ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @sqrt_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.sqrt ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @rsqrt_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.rsqrt ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @rsqrt_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.rsqrt ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @square_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.square ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @square_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.square ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @tanh_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.tanh ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @tanh_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.tanh ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @erf_type_cast(%arg: memref<4x8x16xf16>, %out: memref<4x8x16xf32>) {
  // CHECK: operand 1 ('f16') doesn't match the element type of the enclosing linalg.generic op ('f32')
  linalg.erf ins(%arg : memref<4x8x16xf16>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @erf_broadcast(%arg: memref<8x16xf32>, %out: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.erf ins(%arg : memref<8x16xf32>) outs(%out: memref<4x8x16xf32>)
  return
}

// -----

func.func @max_type_cast(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf16>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op requires the same type for all operands and results
  linalg.max ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf16>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @max_broadcast(%arg0: memref<8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.max ins(%arg0, %arg1 : memref<8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @min_type_cast(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf16>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op requires the same type for all operands and results
  linalg.min ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf16>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @min_broadcast(%arg0: memref<8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.min ins(%arg0, %arg1 : memref<8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @powf_type_cast(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf16>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op requires the same type for all operands and results
  linalg.powf ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf16>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @powf_broadcast(%arg0: memref<8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: op expected operand rank (2) to match the result rank of indexing_map #0 (3)
  linalg.powf ins(%arg0, %arg1 : memref<8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

func.func @select_type_cast(%arg0: memref<4x8x16xi1>, %arg1: memref<4x8x16xf16>, %arg2: memref<4x8x16xf32>, %arg3: memref<4x8x16xf32>) {
  // CHECK: op failed to verify that all of {true_value, false_value, result} have same type
  linalg.select ins(%arg0, %arg1, %arg2 : memref<4x8x16xi1>, memref<4x8x16xf16>, memref<4x8x16xf32>) outs(%arg3: memref<4x8x16xf32>)
  return
}

// -----

func.func @select_wrong_condition_type(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>, %arg3: memref<4x8x16xf32>) {
  // CHECK: op operand #0 must be bool-like, but got 'f32'
  linalg.select ins(%arg0, %arg1, %arg2 : memref<4x8x16xf32>, memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg3: memref<4x8x16xf32>)
  return
}

