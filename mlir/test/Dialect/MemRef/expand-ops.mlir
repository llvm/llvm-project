// RUN: mlir-opt -memref-expand %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @atomic_rmw_to_generic
// CHECK-SAME: ([[F:%.*]]: memref<10xf32>, [[f:%.*]]: f32, [[i:%.*]]: index)
func.func @atomic_rmw_to_generic(%F: memref<10xf32>, %f: f32, %i: index) -> f32 {
  %x = memref.atomic_rmw maximumf %f, %F[%i] : (f32, memref<10xf32>) -> f32
  %y = memref.atomic_rmw minimumf %f, %F[%i] : (f32, memref<10xf32>) -> f32
  return %x : f32
}
// CHECK: [[RESULT:%.*]] = memref.generic_atomic_rmw %arg0[%arg2] : memref<10xf32> {
// CHECK: ^bb0([[CUR_VAL:%.*]]: f32):
// CHECK:   [[MAXIMUM:%.*]] = arith.maximumf [[CUR_VAL]], [[f]] : f32
// CHECK:   memref.atomic_yield [[MAXIMUM]] : f32
// CHECK: }
// CHECK: memref.generic_atomic_rmw %arg0[%arg2] : memref<10xf32> {
// CHECK: ^bb0([[CUR_VAL:%.*]]: f32):
// CHECK:   [[MINIMUM:%.*]] = arith.minimumf [[CUR_VAL]], [[f]] : f32
// CHECK:   memref.atomic_yield [[MINIMUM]] : f32
// CHECK: }
// CHECK: return [[RESULT]] : f32

// -----

// CHECK-LABEL: func @atomic_rmw_no_conversion
func.func @atomic_rmw_no_conversion(%F: memref<10xf32>, %f: f32, %i: index) -> f32 {
  %x = memref.atomic_rmw addf %f, %F[%i] : (f32, memref<10xf32>) -> f32
  return %x : f32
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

// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[C8:%.*]] = arith.constant 8 : index
// CHECK: [[STRIDE_1:%.*]] = arith.muli [[C1]], [[C8]] : index

// CHECK: [[C1_:%.*]] = arith.constant 1 : index
// CHECK: [[DIM_1:%.*]] = memref.load [[SHAPE]]{{\[}}[[C1_]]] : memref<3xi32>
// CHECK: [[SIZE_1:%.*]] = arith.index_cast [[DIM_1]] : i32 to index
// CHECK: [[STRIDE_0:%.*]] = arith.muli [[STRIDE_1]], [[SIZE_1]] : index

// CHECK: [[C0:%.*]] = arith.constant 0 : index
// CHECK: [[DIM_0:%.*]] = memref.load [[SHAPE]]{{\[}}[[C0]]] : memref<3xi32>
// CHECK: [[SIZE_0:%.*]] = arith.index_cast [[DIM_0]] : i32 to index

// CHECK: [[RESULT:%.*]] = memref.reinterpret_cast [[SRC]]
// CHECK-SAME: to offset: [0], sizes: {{\[}}[[SIZE_0]], [[SIZE_1]], 8],
// CHECK-SAME: strides: {{\[}}[[STRIDE_0]], [[STRIDE_1]], [[C1]]]
// CHECK-SAME: : memref<*xf32> to memref<?x?x8xf32>
