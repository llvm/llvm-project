// RUN: mlir-opt %s -sparse-tensor-codegen | FileCheck %s

#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

// This test verifies that sparse_tensor.expand codegen handles the case where
// the input sparse tensor is a function block argument.
// CHECK-LABEL: func.func @sparse_expansion(
// CHECK-SAME: %[[A0:.*]]: memref<?xindex>, %[[A1:.*]]: memref<?xindex>, %[[A2:.*]]: memref<?xindex>, %[[A3:.*]]: memref<?xindex>, %[[A4:.*]]: memref<?xf64>, %[[A5:.*]]: !sparse_tensor.storage_specifier<#sparse>) -> index
// CHECK-NOT: sparse_tensor.expand

module {
  func.func @sparse_expansion(%arg0: tensor<8x8xf64, #sparse>) -> index {
    %values, %filled, %added, %count = sparse_tensor.expand %arg0
      : tensor<8x8xf64, #sparse> to memref<?xf64>, memref<?xi1>, memref<?xindex>
    return %count : index
  }
}
