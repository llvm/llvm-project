// RUN: mlir-opt %s -slice-analysis-test -split-input-file | mlir-query -c "match getDefinitions(hasOpName(\"memref.dealloc\"), 2, false, true, true).extract(\"backward_slice\")" | FileCheck %s

func.func @slicing_linalg_op(%arg0 : index, %arg1 : index, %arg2 : index) {
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

// CHECK: func.func @backward_slice(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: index, 
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index, 
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) {
// CHECK:        %[[ALLOC:.*]] = memref.alloc(%[[ARG1]], %[[ARG2]]) : memref<?x?xf32>
// CHECK-NEXT:   %[[ALLOC_0:.*]] = memref.alloc(%[[ARG0]], %[[ARG2]]) : memref<?x?xf32>
// CHECK-NEXT:   %[[ALLOC_1:.*]] = memref.alloc(%[[ARG1]], %[[ARG0]]) : memref<?x?xf32> 
// CHECK-NEXT:   %[[ALLOC_2:.*]] = memref.alloc(%[[ARG1]], %[[ARG2]]) : memref<?x?xf32>
// CHECK-NEXT:   return %[[ALLOC]], %[[ALLOC_0]], %[[ALLOC_1]], %[[ALLOC_2]] : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
