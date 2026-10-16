// RUN: mlir-opt -memref-expand %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @atomic_rmw_no_conversion
func.func @atomic_rmw_no_conversion(%F: memref<10xf32>, %f: f32, %i: index) -> (f32, f32) {
  %x = memref.atomic_rmw addf %f, %F[%i] : (f32, memref<10xf32>) -> f32
  %y = memref.atomic_rmw maximumf %f, %F[%i] : (f32, memref<10xf32>) -> f32
  return %x, %y : f32, f32
}
// CHECK-NOT: generic_atomic_rmw

// -----

// CHECK-LABEL: func @memref_reshape(
func.func @memref_reshape(%input: memref<*xf32>,
                     %shape: memref<3xi32>) -> memref<?x?x8xf32> {
  %result = memref.reshape %input(%shape)
               : (memref<*xf32>, memref<3xi32>) -> memref<?x?x8xf32>
  return %result : memref<?x?x8xf32>
}
// CHECK-SAME: [[SRC:%.*]]: memref<*xf32>,
// CHECK-SAME: [[SHAPE:%.*]]: memref<3xi32>) -> memref<?x?x8xf32> {

// CHECK: [[C8:%.*]] = arith.constant 8 : index
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[DIM_1:%.*]] = memref.load [[SHAPE]]{{\[}}[[C1]]] : memref<3xi32>
// CHECK: [[SIZE_1:%.*]] = arith.index_cast [[DIM_1]] : i32 to index

// CHECK: [[C8_:%.*]] = arith.constant 8 : index
// CHECK: [[STRIDE_0:%.*]] = arith.muli [[C8_]], [[SIZE_1]] : index

// CHECK: [[C0:%.*]] = arith.constant 0 : index
// CHECK: [[DIM_0:%.*]] = memref.load [[SHAPE]]{{\[}}[[C0]]] : memref<3xi32>
// CHECK: [[SIZE_0:%.*]] = arith.index_cast [[DIM_0]] : i32 to index

// CHECK: [[RESULT:%.*]] = memref.reinterpret_cast [[SRC]]
// CHECK-SAME: to offset: [0], sizes: {{\[}}[[SIZE_0]], [[SIZE_1]], 8],
// CHECK-SAME: strides: {{\[}}[[STRIDE_0]], 8, 1]
// CHECK-SAME: : memref<*xf32> to memref<?x?x8xf32>
