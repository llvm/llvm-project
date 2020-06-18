// RUN: mlir-opt %s -expand-atomic -split-input-file | FileCheck %s

// CHECK-LABEL: func @atomic_rmw_to_generic
// CHECK-SAME: ([[F:%.*]]: memref<10xf32>, [[f:%.*]]: f32, [[i:%.*]]: index)
func @atomic_rmw_to_generic(%F: memref<10xf32>, %f: f32, %i: index) -> f32 {
  %x = atomic_rmw "maxf" %f, %F[%i] : (f32, memref<10xf32>) -> f32
  return %x : f32
}
// CHECK: %0 = std.generic_atomic_rmw %arg0[%arg2] : memref<10xf32> {
// CHECK: ^bb0([[CUR_VAL:%.*]]: f32):
// CHECK:   [[CMP:%.*]] = cmpf "ogt", [[CUR_VAL]], [[f]] : f32
// CHECK:   [[SELECT:%.*]] = select [[CMP]], [[CUR_VAL]], [[f]] : f32
// CHECK:   atomic_yield [[SELECT]] : f32
// CHECK: }
// CHECK: return %0 : f32

// -----

// CHECK-LABEL: func @atomic_rmw_no_conversion
func @atomic_rmw_no_conversion(%F: memref<10xf32>, %f: f32, %i: index) -> f32 {
  %x = atomic_rmw "addf" %f, %F[%i] : (f32, memref<10xf32>) -> f32
  return %x : f32
}
// CHECK-NOT: generic_atomic_rmw
