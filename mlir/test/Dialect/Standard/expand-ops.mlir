// RUN: mlir-opt -std-expand %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @atomic_rmw_to_generic
// CHECK-SAME: ([[F:%.*]]: memref<10xf32>, [[f:%.*]]: f32, [[i:%.*]]: index)
func @atomic_rmw_to_generic(%F: memref<10xf32>, %f: f32, %i: index) -> f32 {
  %x = atomic_rmw maxf %f, %F[%i] : (f32, memref<10xf32>) -> f32
  return %x : f32
}
// CHECK: %0 = std.generic_atomic_rmw %arg0[%arg2] : memref<10xf32> {
// CHECK: ^bb0([[CUR_VAL:%.*]]: f32):
// CHECK:   [[CMP:%.*]] = cmpf ogt, [[CUR_VAL]], [[f]] : f32
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

// Test ceil divide with signed integer
// CHECK-LABEL:       func @ceildivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func @ceildivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = ceildivi_signed %arg0, %arg1 : i32
  return %res : i32

// CHECK:           [[ONE:%.+]] = constant 1 : i32
// CHECK:           [[ZERO:%.+]] = constant 0 : i32
// CHECK:           [[MINONE:%.+]] = constant -1 : i32
// CHECK:           [[CMP1:%.+]] = cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[X:%.+]] = select [[CMP1]], [[MINONE]], [[ONE]] : i32
// CHECK:           [[TRUE1:%.+]] = addi [[X]], [[ARG0]] : i32
// CHECK:           [[TRUE2:%.+]] = divi_signed [[TRUE1]], [[ARG1]] : i32
// CHECK:           [[TRUE3:%.+]] = addi [[ONE]], [[TRUE2]] : i32
// CHECK:           [[FALSE1:%.+]] = subi [[ZERO]], [[ARG0]] : i32
// CHECK:           [[FALSE2:%.+]] = divi_signed [[FALSE1]], [[ARG1]] : i32
// CHECK:           [[FALSE3:%.+]] = subi [[ZERO]], [[FALSE2]] : i32
// CHECK:           [[NNEG:%.+]] = cmpi slt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[NPOS:%.+]] = cmpi sgt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[MPOS:%.+]] = cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[TERM1:%.+]] = and [[NNEG]], [[MNEG]] : i1
// CHECK:           [[TERM2:%.+]] = and [[NPOS]], [[MPOS]] : i1
// CHECK:           [[CMP2:%.+]] = or [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = select [[CMP2]], [[TRUE3]], [[FALSE3]] : i32
}

// -----

// Test floor divide with signed integer
// CHECK-LABEL:       func @floordivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func @floordivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = floordivi_signed %arg0, %arg1 : i32
  return %res : i32
// CHECK:           [[ONE:%.+]] = constant 1 : i32
// CHECK:           [[ZERO:%.+]] = constant 0 : i32
// CHECK:           [[MIN1:%.+]] = constant -1 : i32
// CHECK:           [[CMP1:%.+]] = cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[X:%.+]] = select [[CMP1]], [[ONE]], [[MIN1]] : i32
// CHECK:           [[TRUE1:%.+]] = subi [[X]], [[ARG0]] : i32
// CHECK:           [[TRUE2:%.+]] = divi_signed [[TRUE1]], [[ARG1]] : i32
// CHECK:           [[TRUE3:%.+]] = subi [[MIN1]], [[TRUE2]] : i32
// CHECK:           [[FALSE:%.+]] = divi_signed [[ARG0]], [[ARG1]] : i32
// CHECK:           [[NNEG:%.+]] = cmpi slt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[NPOS:%.+]] = cmpi sgt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[MPOS:%.+]] = cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[TERM1:%.+]] = and [[NNEG]], [[MPOS]] : i1
// CHECK:           [[TERM2:%.+]] = and [[NPOS]], [[MNEG]] : i1
// CHECK:           [[CMP2:%.+]] = or [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = select [[CMP2]], [[TRUE3]], [[FALSE]] : i32
}

// -----

// CHECK-LABEL: func @memref_reshape(
func @memref_reshape(%input: memref<*xf32>,
                     %shape: memref<3xi32>) -> memref<?x?x?xf32> {
  %result = memref.reshape %input(%shape)
               : (memref<*xf32>, memref<3xi32>) -> memref<?x?x?xf32>
  return %result : memref<?x?x?xf32>
}
// CHECK-SAME: [[SRC:%.*]]: memref<*xf32>,
// CHECK-SAME: [[SHAPE:%.*]]: memref<3xi32>) -> memref<?x?x?xf32> {

// CHECK: [[C1:%.*]] = constant 1 : index
// CHECK: [[C2:%.*]] = constant 2 : index
// CHECK: [[DIM_2:%.*]] = memref.load [[SHAPE]]{{\[}}[[C2]]] : memref<3xi32>
// CHECK: [[SIZE_2:%.*]] = index_cast [[DIM_2]] : i32 to index
// CHECK: [[STRIDE_1:%.*]] = muli [[C1]], [[SIZE_2]] : index

// CHECK: [[C1_:%.*]] = constant 1 : index
// CHECK: [[DIM_1:%.*]] = memref.load [[SHAPE]]{{\[}}[[C1_]]] : memref<3xi32>
// CHECK: [[SIZE_1:%.*]] = index_cast [[DIM_1]] : i32 to index
// CHECK: [[STRIDE_0:%.*]] = muli [[STRIDE_1]], [[SIZE_1]] : index

// CHECK: [[C0:%.*]] = constant 0 : index
// CHECK: [[DIM_0:%.*]] = memref.load [[SHAPE]]{{\[}}[[C0]]] : memref<3xi32>
// CHECK: [[SIZE_0:%.*]] = index_cast [[DIM_0]] : i32 to index

// CHECK: [[RESULT:%.*]] = memref.reinterpret_cast [[SRC]]
// CHECK-SAME: to offset: [0], sizes: {{\[}}[[SIZE_0]], [[SIZE_1]], [[SIZE_2]]],
// CHECK-SAME: strides: {{\[}}[[STRIDE_0]], [[STRIDE_1]], [[C1]]]
// CHECK-SAME: : memref<*xf32> to memref<?x?x?xf32>
