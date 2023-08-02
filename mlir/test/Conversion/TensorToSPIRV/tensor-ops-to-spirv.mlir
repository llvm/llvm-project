// RUN: mlir-opt --split-input-file --convert-tensor-to-spirv \
// RUN:   --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// tensor.extract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @tensor_extract_constant
// CHECK-SAME: (%[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32)
func.func @tensor_extract_constant(%a : index, %b: index, %c: index) -> i32 {
  // CHECK: %[[CST:.+]] = spirv.Constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]>
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  // CHECK: %[[VAR:.+]] = spirv.Variable : !spirv.ptr<!spirv.array<12 x i32>, Function>
  // CHECK: spirv.Store "Function" %[[VAR]], %[[CST]] : !spirv.array<12 x i32>
  // CHECK: %[[C0:.+]] = spirv.Constant 0 : i32
  // CHECK: %[[C6:.+]] = spirv.Constant 6 : i32
  // CHECK: %[[MUL0:.+]] = spirv.IMul %[[C6]], %[[A]] : i32
  // CHECK: %[[ADD0:.+]] = spirv.IAdd %[[C0]], %[[MUL0]] : i32
  // CHECK: %[[C3:.+]] = spirv.Constant 3 : i32
  // CHECK: %[[MUL1:.+]] = spirv.IMul %[[C3]], %[[B]] : i32
  // CHECK: %[[ADD1:.+]] = spirv.IAdd %[[ADD0]], %[[MUL1]] : i32
  // CHECK: %[[C1:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[MUL2:.+]] = spirv.IMul %[[C1]], %[[C]] : i32
  // CHECK: %[[ADD2:.+]] = spirv.IAdd %[[ADD1]], %[[MUL2]] : i32
  // CHECK: %[[AC:.+]] = spirv.AccessChain %[[VAR]][%[[ADD2]]]
  // CHECK: %[[VAL:.+]] = spirv.Load "Function" %[[AC]] : i32
  %extract = tensor.extract %cst[%a, %b, %c] : tensor<2x2x3xi32>
  // CHECK: spirv.ReturnValue %[[VAL]]
  return %extract : i32
}

// -----

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @tensor_0d
// CHECK-NEXT:    spirv.Constant 1 : i32
func.func @tensor_0d() -> () {
  %x = arith.constant dense<1> : tensor<i32>
  return
}

// CHECK-LABEL: func @tensor_1d
// CHECK-NEXT:    spirv.Constant dense<[1, 2, 3]> : tensor<3xi32> : !spirv.array<3 x i32>
func.func @tensor_1d() -> () {
  %x = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  return
}

// CHECK-LABEL: func @tensor_2d
// CHECK-NEXT:    spirv.Constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spirv.array<6 x i32>
func.func @tensor_2d() -> () {
  %x = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  return
}

// We do not handle zero-element tensors yet. Just make we do not crash on them.
// CHECK-LABEL: func @tensor_2d_empty
// CHECK-NEXT:    arith.constant dense<>
func.func @tensor_2d_empty() -> () {
  %x = arith.constant dense<> : tensor<2x0xi32>
  return
}
