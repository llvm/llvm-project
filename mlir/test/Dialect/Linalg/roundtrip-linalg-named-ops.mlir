// RUN: mlir-opt %s -linalg-generalize-named-ops | mlir-opt --linalg-specialize-generic-ops | FileCheck %s

func.func @roundtrip_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @roundtrip_matmul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

func.func @roundtrip_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.add ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: roundtrip_add
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,  %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

func.func @roundtrip_exp(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.exp ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK-LABEL: roundtrip_exp
// CHECK-SAME: %[[A:.+]]: memref<7x14x21xf32>, %[[Out:.+]]: memref<7x14x21xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.exp ins(%[[A]] : memref<7x14x21xf32>) outs(%[[Out]] : memref<7x14x21xf32>)

// -----

func.func @roundtrip_gemm(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.add ins(%0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @roundtrip_gemm
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[C:.+]]: tensor<?x?xf32>, %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: %[[AB:.+]] = linalg.matmul ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: linalg.add ins(%[[AB]], %[[C]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>
