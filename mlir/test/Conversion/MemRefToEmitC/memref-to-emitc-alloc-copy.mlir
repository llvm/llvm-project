// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s --check-prefixes=CPP,CHECK
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s -split-input-file | FileCheck %s --check-prefixes=NOCPP,CHECK

func.func @alloc_copy(%arg0: memref<999xi32>) {
  %alloc = memref.alloc() : memref<999xi32>
  memref.copy %arg0, %alloc : memref<999xi32> to memref<999xi32>
  %alloc_1 = memref.alloc() : memref<999xi32>
  memref.copy %arg0, %alloc_1 : memref<999xi32> to memref<999xi32>
  return
} 

// NOCPP:  emitc.include <"stdlib.h">
// NOCPP-NEXT:  emitc.include <"string.h">

// CPP:  emitc.include <"cstdlib">
// CPP-NEXT:  emitc.include <"cstring">

// CHECK-LABEL:   func.func @alloc_copy(
// CHECK-SAME:      %[[ARG0:.*]]: memref<999xi32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<999xi32> to !emitc.array<999xi32>
// CHECK:           %[[CALL_OPAQUE_0:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK:           %[[VAL_0:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK:           %[[MUL_0:.*]] = emitc.mul %[[CALL_OPAQUE_0]], %[[VAL_0]] : (!emitc.size_t, index) -> !emitc.size_t
// CHECK:           %[[CALL_OPAQUE_1:.*]] = emitc.call_opaque "malloc"(%[[MUL_0]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK:           %[[CAST_0:.*]] = emitc.cast %[[CALL_OPAQUE_1]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[CAST_0]] : !emitc.ptr<i32> to !emitc.array<999xi32>
// CHECK:           %[[VAL_1:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[SUBSCRIPT_0:.*]] = emitc.subscript %[[UNREALIZED_CONVERSION_CAST_0]]{{\[}}%[[VAL_1]]] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32>
// CHECK:           %[[APPLY_0:.*]] = emitc.apply "&"(%[[SUBSCRIPT_0]]) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[SUBSCRIPT_1:.*]] = emitc.subscript %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_2]]] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32>
// CHECK:           %[[APPLY_1:.*]] = emitc.apply "&"(%[[SUBSCRIPT_1]]) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK:           %[[CALL_OPAQUE_2:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK:           %[[VAL_3:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK:           %[[MUL_1:.*]] = emitc.mul %[[CALL_OPAQUE_2]], %[[VAL_3]] : (!emitc.size_t, index) -> !emitc.size_t
// CHECK:           emitc.call_opaque "memcpy"(%[[APPLY_1]], %[[APPLY_0]], %[[MUL_1]]) : (!emitc.ptr<i32>, !emitc.ptr<i32>, !emitc.size_t) -> ()
// CHECK:           %[[CALL_OPAQUE_3:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK:           %[[VAL_4:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK:           %[[MUL_2:.*]] = emitc.mul %[[CALL_OPAQUE_3]], %[[VAL_4]] : (!emitc.size_t, index) -> !emitc.size_t
// CHECK:           %[[CALL_OPAQUE_4:.*]] = emitc.call_opaque "malloc"(%[[MUL_2]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK:           %[[CAST_1:.*]] = emitc.cast %[[CALL_OPAQUE_4]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[CAST_1]] : !emitc.ptr<i32> to !emitc.array<999xi32>
// CHECK:           %[[VAL_5:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[SUBSCRIPT_2:.*]] = emitc.subscript %[[UNREALIZED_CONVERSION_CAST_0]]{{\[}}%[[VAL_5]]] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32>
// CHECK:           %[[APPLY_2:.*]] = emitc.apply "&"(%[[SUBSCRIPT_2]]) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[SUBSCRIPT_3:.*]] = emitc.subscript %[[UNREALIZED_CONVERSION_CAST_2]]{{\[}}%[[VAL_6]]] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32>
// CHECK:           %[[APPLY_3:.*]] = emitc.apply "&"(%[[SUBSCRIPT_3]]) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK:           %[[CALL_OPAQUE_5:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK:           %[[VAL_7:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK:           %[[MUL_3:.*]] = emitc.mul %[[CALL_OPAQUE_5]], %[[VAL_7]] : (!emitc.size_t, index) -> !emitc.size_t
// CHECK:           emitc.call_opaque "memcpy"(%[[APPLY_3]], %[[APPLY_2]], %[[MUL_3]]) : (!emitc.ptr<i32>, !emitc.ptr<i32>, !emitc.size_t) -> ()
// CHECK:           return
// CHECK:         }
