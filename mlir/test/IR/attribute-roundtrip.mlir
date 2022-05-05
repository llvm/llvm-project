// RUN: mlir-opt -canonicalize %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @large_i1_tensor_roundtrip
func @large_i1_tensor_roundtrip() -> tensor<160xi1> {
  %cst_0 = arith.constant dense<"0xFFF00000FF000000FF000000FF000000FF000000"> : tensor<160xi1>
  %cst_1 = arith.constant dense<"0xFF000000FF000000FF000000FF000000FF0000F0"> : tensor<160xi1>
  // CHECK: dense<"0xFF000000FF000000FF000000FF000000FF000000">
  %0 = arith.andi %cst_0, %cst_1 : tensor<160xi1>
  return %0 : tensor<160xi1>
}
