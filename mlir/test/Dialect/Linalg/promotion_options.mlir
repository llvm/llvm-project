// RUN: mlir-opt %s -test-linalg-transform-patterns=test-linalg-promotion-options -split-input-file | FileCheck %s

func @gemm(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
   linalg.matmul {__internal_linalg_transform__ = "START"}
     ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
    outs(%c: memref<?x?xf32>)
   return
}

//      CHECK: func @gemm
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG: %[[C42:.+]] = constant 4.200000e+01 : f32
//      CHECK: scf.for
//      CHECK:   scf.for
//      CHECK:     scf.for
//      CHECK:       %[[T7:.+]] = subview %[[ARG0]]
//      CHECK:       %[[T12:.+]] = subview %[[ARG1]]
//      CHECK:       %[[T17:.+]] = subview %[[ARG2]]
//      CHECK:       %[[T18:.+]] = memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32, 3>
//      CHECK:       %[[T19:.+]] = subview %[[T18]]
//      CHECK:       %[[T20:.+]] = memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32, 3>
//      CHECK:       %[[T21:.+]] = subview %[[T20]]
//      CHECK:       linalg.fill(%[[T19]], %[[C42]])
//      CHECK:       linalg.copy(%[[T7]], %[[T19]])
//      CHECK:       linalg.fill(%[[T21]], %[[C42]])
//      CHECK:       linalg.copy(%[[T17]], %[[T21]])
//      CHECK:       linalg.matmul ins(%[[T19]], %[[T12]]{{.*}} outs(%[[T21]]
//  CHECK-NOT:       linalg.fill
//      CHECK:       linalg.copy(%[[T21]], %[[T17]])
//      CHECK:       memref.dealloc %[[T18]]
//      CHECK:       memref.dealloc %[[T20]]
