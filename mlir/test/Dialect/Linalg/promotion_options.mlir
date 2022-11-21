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
//      CHECK:       %[[svA:.+]] = memref.subview %[[ARG0]]
//      CHECK:       %[[svB:.+]] = memref.subview %[[ARG1]]
//      CHECK:       %[[svC:.+]] = memref.subview %[[ARG2]]

//      CHECK:       %[[tmpA:.*]] = memref.alloc() : memref<1024xi8>
//      CHECK:       %[[VA:.*]] = memref.view %[[tmpA]][%[[C0]]][] : memref<1024xi8> to memref<16x16xf32>
//      CHECK:       %[[svAA:.+]] = memref.subview %[[VA]]

//      CHECK:       %[[tmpC:.*]] = memref.alloc() : memref<1024xi8>
//      CHECK:       %[[VC:.*]] = memref.view %[[tmpC]][%[[C0]]][] : memref<1024xi8> to memref<16x16xf32>
//      CHECK:       %[[svCC:.+]] = memref.subview %[[VC]]

//      CHECK:       memref.copy %[[svA]], %[[svAA]]
//      CHECK:       memref.copy %[[svC]], %[[svCC]]
//      CHECK:       linalg.matmul ins(%[[VA]], %[[svB]]{{.*}} outs(%[[VC]]
//      CHECK:       memref.copy %[[svCC]], %[[svC]]
//      CHECK:       memref.dealloc %[[tmpA]]
//      CHECK:       memref.dealloc %[[tmpC]]

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
  %1, %loops:3 = transform.structured.tile %0 [16, 16, 16]
  %2 = transform.structured.promote %1 { operands_to_promote = [0, 2], force_full_tiles = [false, false], use_full_tiles_by_default }
}
