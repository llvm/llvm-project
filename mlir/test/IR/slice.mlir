// RUN: mlir-opt -slice-analysis-test %s | FileCheck %s

func @slicing_linalg_op(%arg0 : index, %arg1 : index, %arg2 : index) {
  %a = alloc(%arg0, %arg2) : memref<?x?xf32>
  %b = alloc(%arg2, %arg1) : memref<?x?xf32>
  %c = alloc(%arg0, %arg1) : memref<?x?xf32>
  %d = alloc(%arg0, %arg1) : memref<?x?xf32>
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
               outs(%c : memref<?x?xf32>)
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
               outs(%d : memref<?x?xf32>)
  dealloc %c : memref<?x?xf32>
  dealloc %b : memref<?x?xf32>
  dealloc %a : memref<?x?xf32>
  dealloc %d : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @slicing_linalg_op__backward_slice__0
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[A:.+]] = alloc(%[[ARG0]], %[[ARG2]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[B:.+]] = alloc(%[[ARG2]], %[[ARG1]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[C:.+]] = alloc(%[[ARG0]], %[[ARG1]]) : memref<?x?xf32>
//       CHECK:   return

// CHECK-LABEL: func @slicing_linalg_op__backward_slice__1
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[A:.+]] = alloc(%[[ARG0]], %[[ARG2]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[B:.+]] = alloc(%[[ARG2]], %[[ARG1]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[C:.+]] = alloc(%[[ARG0]], %[[ARG1]]) : memref<?x?xf32>
//       CHECK:   return
