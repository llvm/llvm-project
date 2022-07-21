// RUN: mlir-opt %s -test-transform-dialect-interpreter -canonicalize -split-input-file | FileCheck %s

func.func @gemm(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
   linalg.matmul ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
               outs(%c: memref<?x?xf32>)
   return
}

//      CHECK: func @gemm
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//      CHECK: scf.for
//      CHECK:   scf.for
//      CHECK:     scf.for
//      CHECK:       %[[T7:.+]] = memref.subview %[[ARG0]]
//      CHECK:       %[[T12:.+]] = memref.subview %[[ARG1]]
//      CHECK:       %[[T17:.+]] = memref.subview %[[ARG2]]
//      CHECK:       %[[A0:.*]] = memref.alloc() : memref<1024xi8>
//      CHECK:       %[[V0:.*]] = memref.view %[[A0]][%[[C0]]][] : memref<1024xi8> to memref<16x16xf32>
//      CHECK:       %[[T19:.+]] = memref.subview %[[V0]]
//      CHECK:       %[[A1:.*]] = memref.alloc() : memref<1024xi8>
//      CHECK:       %[[V1:.*]] = memref.view %[[A1]][%[[C0]]][] : memref<1024xi8> to memref<16x16xf32>
//      CHECK:       %[[T21:.+]] = memref.subview %[[V1]]
//      CHECK:       memref.copy %[[T7]], %[[T19]]
//      CHECK:       memref.copy %[[T17]], %[[T21]]
//      CHECK:       linalg.matmul ins(%[[T19]], %[[T12]]{{.*}} outs(%[[T21]]
//      CHECK:       memref.copy %[[T21]], %[[T17]]
//      CHECK:       memref.dealloc %[[A0]]
//      CHECK:       memref.dealloc %[[A1]]

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      %1, %loops:3 = transform.structured.tile %0 [16, 16, 16]
      %2 = transform.structured.promote %1 { operands_to_promote = [0, 2], force_full_tiles = [false, false] }
  }
}
