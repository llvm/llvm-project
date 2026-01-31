// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

module {
  func.func @fold_from_elements_i32() -> tensor<2xi32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %t = tensor.from_elements %c0, %c1 : tensor<2xi32>
    return %t : tensor<2xi32>
  }
}

// CHECK-LABEL: func.func @fold_from_elements_i32
// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1]> : tensor<2xi32>
// CHECK: return %[[CST]] : tensor<2xi32>

// -----

module {
  func.func @fold_from_elements_index() -> tensor<2xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %t = tensor.from_elements %c0, %c1 : tensor<2xindex>
    return %t : tensor<2xindex>
  }
}

// CHECK-LABEL: func.func @fold_from_elements_index
// CHECK: arith.constant dense<[0, 1]> : tensor<2xindex>

