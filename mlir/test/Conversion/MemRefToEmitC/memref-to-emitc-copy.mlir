// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s --check-prefixes=CPP,CHECK
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s -split-input-file | FileCheck %s --check-prefixes=NOCPP,CHECK

func.func @copying(%arg0 : memref<9x4x5x7xf32>, %arg1 : memref<9x4x5x7xf32>) {
  memref.copy %arg0, %arg1 : memref<9x4x5x7xf32> to memref<9x4x5x7xf32>
  return
}

// CHECK: module {
// NOCPP:  emitc.include <"string.h">
// CPP:  emitc.include <"cstring">

// CHECK-LABEL:   func.func @copying(
// CHECK-SAME:      %[[ARG0:.*]]: memref<9x4x5x7xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<9x4x5x7xf32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : memref<9x4x5x7xf32> to !emitc.array<9x4x5x7xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<9x4x5x7xf32> to !emitc.array<9x4x5x7xf32>
// CHECK:           %[[VAL_0:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[SUBSCRIPT_0:.*]] = emitc.subscript %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_0]], %[[VAL_0]], %[[VAL_0]], %[[VAL_0]]] : (!emitc.array<9x4x5x7xf32>, index, index, index, index) -> !emitc.lvalue<f32>
// CHECK:           %[[ADDRESS_OF_0:.*]] = emitc.address_of %[[SUBSCRIPT_0]] : !emitc.lvalue<f32>
// CHECK:           %[[VAL_1:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[SUBSCRIPT_1:.*]] = emitc.subscript %[[UNREALIZED_CONVERSION_CAST_0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_1]], %[[VAL_1]]] : (!emitc.array<9x4x5x7xf32>, index, index, index, index) -> !emitc.lvalue<f32>
// CHECK:           %[[ADDRESS_OF_1:.*]] = emitc.address_of %[[SUBSCRIPT_1]] : !emitc.lvalue<f32>
// CHECK:           %[[CALL_OPAQUE_0:.*]] = emitc.call_opaque "sizeof"() <{args = [f32]}> : () -> !emitc.size_t
// CHECK:           %[[VAL_2:.*]] = "emitc.constant"() <{value = 1260 : index}> : () -> index
// CHECK:           %[[MUL_0:.*]] = emitc.mul %[[CALL_OPAQUE_0]], %[[VAL_2]] : (!emitc.size_t, index) -> !emitc.size_t
// CHECK:           emitc.call_opaque "memcpy"(%[[ADDRESS_OF_1]], %[[ADDRESS_OF_0]], %[[MUL_0]]) : (!emitc.ptr<f32>, !emitc.ptr<f32>, !emitc.size_t) -> ()
// CHECK:           return
// CHECK:         }

// -----

func.func @copying_rank0(%arg0 : memref<i32>, %arg1 : memref<i32>) {
  memref.copy %arg0, %arg1 : memref<i32> to memref<i32>
  return
}

// CHECK: module {

// CHECK-LABEL:   func.func @copying_rank0(
// CHECK-SAME:      %[[ARG0:.*]]: memref<i32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<i32>) {
// CHECK:           %[[TARGET_CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : memref<i32> to !emitc.ptr<i32>
// CHECK:           %[[SOURCE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<i32> to !emitc.ptr<i32>
// CHECK:           %[[ZERO:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[SOURCE_LVALUE:.*]] = emitc.subscript %[[SOURCE_CAST]]{{\[}}%[[ZERO]]] : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
// CHECK:           %[[VALUE:.*]] = emitc.load %[[SOURCE_LVALUE]] : <i32>
// CHECK:           %[[TARGET_LVALUE:.*]] = emitc.subscript %[[TARGET_CAST]]{{\[}}%[[ZERO]]] : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VALUE]] : i32 to %[[TARGET_LVALUE]] : <i32>
// CHECK:           return
// CHECK:         }
