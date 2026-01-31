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

// -----

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
  func.func @from_elements_no_fold_with_poison() -> tensor<2xi32> {
    %p = ub.poison : i32
    %c0 = arith.constant 0 : i32
    %t = tensor.from_elements %p, %c0 : tensor<2xi32>
    return %t : tensor<2xi32>
  }
}

// CHECK-LABEL: func.func @from_elements_no_fold_with_poison
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[T:.*]] = tensor.from_elements %[[P]], %[[C0]] : tensor<2xi32>
// CHECK: return %[[T]] : tensor<2xi32>
// CHECK-NOT: arith.constant dense<
