// RUN: mlir-opt -std-expand %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @atomic_rmw_to_generic
// CHECK-SAME: ([[F:%.*]]: memref<10xf32>, [[f:%.*]]: f32, [[i:%.*]]: index)
func @atomic_rmw_to_generic(%F: memref<10xf32>, %f: f32, %i: index) -> f32 {
  %x = atomic_rmw maxf %f, %F[%i] : (f32, memref<10xf32>) -> f32
  return %x : f32
}
// CHECK: %0 = generic_atomic_rmw %arg0[%arg2] : memref<10xf32> {
// CHECK: ^bb0([[CUR_VAL:%.*]]: f32):
// CHECK:   [[CMP:%.*]] = arith.cmpf ogt, [[CUR_VAL]], [[f]] : f32
// CHECK:   [[SELECT:%.*]] = select [[CMP]], [[CUR_VAL]], [[f]] : f32
// CHECK:   atomic_yield [[SELECT]] : f32
// CHECK: }
// CHECK: return %0 : f32

// -----

// CHECK-LABEL: func @atomic_rmw_no_conversion
func @atomic_rmw_no_conversion(%F: memref<10xf32>, %f: f32, %i: index) -> f32 {
  %x = atomic_rmw addf %f, %F[%i] : (f32, memref<10xf32>) -> f32
  return %x : f32
}
// CHECK-NOT: generic_atomic_rmw

// -----

// CHECK-LABEL: func @memref_reshape(
func @memref_reshape(%input: memref<*xf32>,
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

// -----

// CHECK-LABEL: func @maxf
func @maxf(%a: f32, %b: f32) -> f32 {
  %result = maxf(%a, %b): (f32, f32) -> f32
  return %result : f32
}
// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ogt, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-NEXT: %[[RESULT:.*]] = select %[[IS_NAN]], %[[NAN]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @maxf_vector
func @maxf_vector(%a: vector<4xf16>, %b: vector<4xf16>) -> vector<4xf16> {
  %result = maxf(%a, %b): (vector<4xf16>, vector<4xf16>) -> vector<4xf16>
  return %result : vector<4xf16>
}
// CHECK-SAME: %[[LHS:.*]]: vector<4xf16>, %[[RHS:.*]]: vector<4xf16>)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ogt, %[[LHS]], %[[RHS]] : vector<4xf16>
// CHECK-NEXT: %[[SELECT:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]]
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[LHS]], %[[RHS]] : vector<4xf16>
// CHECK-NEXT: %[[NAN:.*]] = arith.constant 0x7E00 : f16
// CHECK-NEXT: %[[SPLAT_NAN:.*]] = splat %[[NAN]] : vector<4xf16>
// CHECK-NEXT: %[[RESULT:.*]] = select %[[IS_NAN]], %[[SPLAT_NAN]], %[[SELECT]]
// CHECK-NEXT: return %[[RESULT]] : vector<4xf16>

// -----

// CHECK-LABEL: func @minf
func @minf(%a: f32, %b: f32) -> f32 {
  %result = minf(%a, %b): (f32, f32) -> f32
  return %result : f32
}
// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf olt, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-NEXT: %[[RESULT:.*]] = select %[[IS_NAN]], %[[NAN]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32


// -----

// CHECK-LABEL: func @maxsi
func @maxsi(%a: i32, %b: i32) -> i32 {
  %result = maxsi(%a, %b): (i32, i32) -> i32
  return %result : i32
}
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi sgt, %[[LHS]], %[[RHS]] : i32

// -----

// CHECK-LABEL: func @minsi
func @minsi(%a: i32, %b: i32) -> i32 {
  %result = minsi(%a, %b): (i32, i32) -> i32
  return %result : i32
}
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i32


// -----

// CHECK-LABEL: func @maxui
func @maxui(%a: i32, %b: i32) -> i32 {
  %result = maxui(%a, %b): (i32, i32) -> i32
  return %result : i32
}
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi ugt, %[[LHS]], %[[RHS]] : i32


// -----

// CHECK-LABEL: func @minui
func @minui(%a: i32, %b: i32) -> i32 {
  %result = minui(%a, %b): (i32, i32) -> i32
  return %result : i32
}
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : i32
