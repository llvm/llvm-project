// RUN: not mlir-opt -split-input-file -verify-diagnostics %s 2>&1 | FileCheck %s

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

// -----

// linalg.select with all-integer operands
func.func @select_all_integer(%arg0: memref<4x8x16xi32>, %arg1: memref<4x8x16xi32>, %arg2: memref<4x8x16xi32>, %arg3: memref<4x8x16xi32>) {
  // CHECK: op operand #0 must be bool-like, but got 'i32'
  linalg.select ins(%arg0, %arg1, %arg2 : memref<4x8x16xi32>, memref<4x8x16xi32>, memref<4x8x16xi32>) outs(%arg3: memref<4x8x16xi32>)
  return
}

// -----

// Regression test: linalg.select with index type operands should emit a
// diagnostic instead of crashing (https://github.com/llvm/llvm-project/issues/179046).
func.func @select_invalid_index_type(%cond: index, %a: index, %b: index,
                                     %out: tensor<1xindex>) -> tensor<1xindex> {
  // CHECK: op operand #0 must be bool-like, but got 'index'
  %0 = linalg.select ins(%cond, %a, %b : index, index, index)
                     outs(%out : tensor<1xindex>) -> tensor<1xindex>
  return %0 : tensor<1xindex>
}

// -----

// linalg.select with an integer (non-i1) condition and floating-point values:
func.func @select_invalid_integer_cond_float_values(%cond: tensor<4xi32>,
    %a: tensor<4xf32>, %b: tensor<4xf32>,
    %out: tensor<4xf32>) -> tensor<4xf32> {
// CHECK: op operand #0 must be bool-like, but got 'i32'
  %0 = linalg.select ins(%cond, %a, %b : tensor<4xi32>, tensor<4xf32>, tensor<4xf32>)
                     outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
