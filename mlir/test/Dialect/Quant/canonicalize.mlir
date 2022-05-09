// RUN: mlir-opt %s -split-input-file -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// -----
// CHECK-LABEL: redundant_scast
func.func @redundant_scast() -> tensor<4xi8> {
  // CHECK-NEXT: arith.constant dense<10> : tensor<4xi8>
  // CHECK-NEXT: return
  %cst = arith.constant dense<5> : tensor<4xi8>
  %1 = "quant.scast"(%cst) : (tensor<4xi8>) -> tensor<4x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %2 = "quant.scast"(%1) : (tensor<4x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<4xi8>
  %3 = arith.addi %2, %2 : tensor<4xi8>
  return %3 : tensor<4xi8>
}

// -----
// CHECK-LABEL: non_redundant_scast
func.func @non_redundant_scast() -> tensor<4x!quant.uniform<u8:f32, 7.812500e-03:128>> {
  // CHECK-NEXT: arith.constant dense<5> : tensor<4xi8>
  // CHECK-NEXT: scast
  // CHECK-NEXT: return
  %cst = arith.constant dense<5> : tensor<4xi8>
  %1 = "quant.scast"(%cst) : (tensor<4xi8>) -> tensor<4x!quant.uniform<u8:f32, 7.812500e-03:128>>
  return %1 : tensor<4x!quant.uniform<u8:f32, 7.812500e-03:128>>
}
