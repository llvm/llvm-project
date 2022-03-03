// RUN: mlir-opt -split-input-file -convert-tensor-to-spirv -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// tensor.extract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @tensor_extract_constant
// CHECK-SAME: (%[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32)
func @tensor_extract_constant(%a : index, %b: index, %c: index) -> i32 {
  // CHECK: %[[CST:.+]] = spv.Constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]>
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  // CHECK: %[[VAR:.+]] = spv.Variable init(%[[CST]]) : !spv.ptr<!spv.array<12 x i32, stride=4>, Function>
  // CHECK: %[[C0:.+]] = spv.Constant 0 : i32
  // CHECK: %[[C6:.+]] = spv.Constant 6 : i32
  // CHECK: %[[MUL0:.+]] = spv.IMul %[[C6]], %[[A]] : i32
  // CHECK: %[[ADD0:.+]] = spv.IAdd %[[C0]], %[[MUL0]] : i32
  // CHECK: %[[C3:.+]] = spv.Constant 3 : i32
  // CHECK: %[[MUL1:.+]] = spv.IMul %[[C3]], %[[B]] : i32
  // CHECK: %[[ADD1:.+]] = spv.IAdd %[[ADD0]], %[[MUL1]] : i32
  // CHECK: %[[C1:.+]] = spv.Constant 1 : i32
  // CHECK: %[[MUL2:.+]] = spv.IMul %[[C1]], %[[C]] : i32
  // CHECK: %[[ADD2:.+]] = spv.IAdd %[[ADD1]], %[[MUL2]] : i32
  // CHECK: %[[AC:.+]] = spv.AccessChain %[[VAR]][%[[ADD2]]]
  // CHECK: %[[VAL:.+]] = spv.Load "Function" %[[AC]] : i32
  %extract = tensor.extract %cst[%a, %b, %c] : tensor<2x2x3xi32>
  // CHECK: spv.ReturnValue %[[VAL]]
  return %extract : i32
}
