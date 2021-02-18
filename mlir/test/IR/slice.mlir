// RUN: mlir-opt -slice-analysis-test %s | FileCheck %s

func @slicing_linalg_op(%arg0 : index, %arg1 : index, %arg2 : index) {
  %a = memref.alloc(%arg0, %arg2) : memref<?x?xf32>
  %b = memref.alloc(%arg2, %arg1) : memref<?x?xf32>
  %c = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %d = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
               outs(%c : memref<?x?xf32>)
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
               outs(%d : memref<?x?xf32>)
  memref.dealloc %c : memref<?x?xf32>
  memref.dealloc %b : memref<?x?xf32>
  memref.dealloc %a : memref<?x?xf32>
  memref.dealloc %d : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @slicing_linalg_op__backward_slice__0
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[A:.+]] = memref.alloc(%[[ARG0]], %[[ARG2]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[B:.+]] = memref.alloc(%[[ARG2]], %[[ARG1]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[C:.+]] = memref.alloc(%[[ARG0]], %[[ARG1]]) : memref<?x?xf32>
//       CHECK:   return

// CHECK-LABEL: func @slicing_linalg_op__backward_slice__1
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[A:.+]] = memref.alloc(%[[ARG0]], %[[ARG2]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[B:.+]] = memref.alloc(%[[ARG2]], %[[ARG1]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[C:.+]] = memref.alloc(%[[ARG0]], %[[ARG1]]) : memref<?x?xf32>
//       CHECK:   return
