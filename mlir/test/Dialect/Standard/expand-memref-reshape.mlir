// RUN: mlir-opt %s -test-expand-memref-reshape | FileCheck %s

// CHECK-LABEL: func @memref_reshape(
func @memref_reshape(%input: memref<*xf32>,
                     %shape: memref<3xi32>) -> memref<?x?x?xf32> {
  %result = memref_reshape %input(%shape)
               : (memref<*xf32>, memref<3xi32>) -> memref<?x?x?xf32>
  return %result : memref<?x?x?xf32>
}
// CHECK-SAME: [[SRC:%.*]]: memref<*xf32>,
// CHECK-SAME: [[SHAPE:%.*]]: memref<3xi32>) -> memref<?x?x?xf32> {
// CHECK: [[C2:%.*]] = constant 2 : index
// CHECK: [[C1:%.*]] = constant 1 : index
// CHECK: [[C0:%.*]] = constant 0 : index
// CHECK: [[DIM_2:%.*]] = load [[SHAPE]]{{\[}}[[C2]]] : memref<3xi32>
// CHECK: [[SIZE_2:%.*]] = index_cast [[DIM_2]] : i32 to index
// CHECK: [[DIM_1:%.*]] = load [[SHAPE]]{{\[}}[[C1]]] : memref<3xi32>
// CHECK: [[SIZE_1:%.*]] = index_cast [[DIM_1]] : i32 to index
// CHECK: [[STRIDE_0:%.*]] = muli [[SIZE_2]], [[SIZE_1]] : index
// CHECK: [[DIM_0:%.*]] = load [[SHAPE]]{{\[}}[[C0]]] : memref<3xi32>
// CHECK: [[SIZE_0:%.*]] = index_cast [[DIM_0]] : i32 to index

// CHECK: [[RESULT:%.*]] = memref_reinterpret_cast [[SRC]]
// CHECK-SAME: to offset: [0], sizes: {{\[}}[[SIZE_0]], [[SIZE_1]], [[SIZE_2]]],
// CHECK-SAME: strides: {{\[}}[[STRIDE_0]], [[SIZE_2]], [[C1]]]
// CHECK-SAME: : memref<*xf32> to memref<?x?x?xf32>
