// RUN: mlir-opt -convert-to-emitc -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Negative tests (must NOT rewrite)
//===----------------------------------------------------------------------===//

func.func private @negative_rank0_alloc() -> memref<i32> {
  %alloc = memref.alloc() : memref<i32>
  return %alloc : memref<i32>
}
// CHECK-LABEL:   func.func private @negative_rank0_alloc() -> memref<i32>
// CHECK:           %[[ALLOC:.*]] = memref.alloc()
// CHECK:           return %[[ALLOC]]

func.func private @negative_rank0_arg(%arg0: memref<i32>) -> memref<i32> {
  return %arg0 : memref<i32>
}
// CHECK-LABEL:   func.func private @negative_rank0_arg(
// CHECK-SAME:      %[[SRC:.*]]: memref<i32>) -> memref<i32>
// CHECK:           return %[[SRC]] : memref<i32>

func.func private @negative_rank1_alloc() -> memref<1xi32> {
  %alloc = memref.alloc() : memref<1xi32>
  return %alloc : memref<1xi32>
}
// CHECK-LABEL:   func.func private @negative_rank1_alloc() -> memref<1xi32>
// CHECK:           %[[SIZEOF_I32:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK:           %[[NUM_ELEMS:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
// CHECK:           %[[TOTAL_BYTES:.*]] = emitc.mul %[[SIZEOF_I32]], %[[NUM_ELEMS]] : (!emitc.size_t, index) -> !emitc.size_t
// CHECK:           %[[MALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[TOTAL_BYTES]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK:           %[[ELEM_PTR:.*]] = emitc.cast %[[MALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK:           %[[UNREALIZED_CONVERSION_ELEM_PTR:.*]] = builtin.unrealized_conversion_cast %[[ELEM_PTR]] : !emitc.ptr<i32> to memref<1xi32>
// CHECK:           return %[[UNREALIZED_CONVERSION_ELEM_PTR]] : memref<1xi32>

func.func private @negative_rank1_arg(%arg0: memref<1xi32>) -> memref<1xi32> {
  return %arg0 : memref<1xi32>
}
// CHECK-LABEL:   func.func private @negative_rank1_arg(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi32>) -> memref<1xi32>
// CHECK:           return %[[SRC]] : memref<1xi32>

func.func private @negative_rank2_arg(%arg0: memref<1x1xi32>) -> memref<1x1xi32> {
  return %arg0 : memref<1x1xi32>
}
// CHECK-LABEL:   func.func private @negative_rank2_arg(
// CHECK-SAME:      %[[SRC:.*]]: memref<1x1xi32>) -> memref<1x1xi32>
// CHECK:           return %[[SRC]] : memref<1x1xi32>

func.func private @negative_rank0_constant() -> memref<i64> {
  %0 = arith.constant dense<-1> : memref<i64>
  return %0 : memref<i64>
}
// CHECK-LABEL:   func.func private @negative_rank0_constant() -> memref<i64> {
// CHECK:           %[[CONST:.*]] = arith.constant dense<-1> : memref<i64>
// CHECK:           return %[[CONST]] : memref<i64>

func.func private @negative_rank1_constant() -> memref<1xi64> {
  %0 = arith.constant dense<[-1]> : memref<1xi64>
  return %0 : memref<1xi64>
}
// CHECK-LABEL:   func.func private @negative_rank1_constant() -> memref<1xi64> {
// CHECK:           %[[CONST:.*]] = arith.constant dense<-1> : memref<1xi64>
// CHECK:           return %[[CONST]] : memref<1xi64>

func.func private @multiple_elements(%arg0: memref<2xi64>) -> memref<2xi64> {
  return %arg0 : memref<2xi64>
}
// CHECK-LABEL:   func.func private @multiple_elements(
// CHECK-SAME:      %[[SRC:.*]]: memref<2xi64>) -> memref<2xi64>
// CHECK:           return %[[SRC]] : memref<2xi64>

func.func private @negative_multiple_returns(%arg0: memref<1xi32>) -> (memref<1xi32>, i32) {
  %1 = arith.constant 7 : i32
  return %arg0, %1 : memref<1xi32>, i32
}
// CHECK-LABEL:   func.func private @negative_multiple_returns(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi32>) -> (memref<1xi32>, i32)
// CHECK:           %[[C0:.*]] = "emitc.constant"() <{value = 7 : i32}> : () -> i32
// CHECK:           return %[[SRC]], %[[C0]] : memref<1xi32>, i32

func.func private @negative_dynamic_shape(%arg0: memref<?xi32>) -> memref<?xi32> {
  return %arg0 : memref<?xi32>
}
// CHECK-LABEL:   func.func private @negative_dynamic_shape(
// CHECK-SAME:      %[[SRC:.*]]: memref<?xi32>) -> memref<?xi32>
// CHECK:           return %[[SRC]] : memref<?xi32>

func.func private @negative_non_identity_layout(%arg0: memref<1x1xi32, strided<[2, 1], offset: 0>>)
    -> memref<1x1xi32, strided<[2, 1], offset: 0>> {
  return %arg0 : memref<1x1xi32, strided<[2, 1], offset: 0>>
}
// CHECK-LABEL:   func.func private @negative_non_identity_layout(
// CHECK-SAME:      %[[SRC:.*]]: memref<1x1xi32, strided<[2, 1]>>) -> memref<1x1xi32, strided<[2, 1]>>
// CHECK:           return %[[SRC]] : memref<1x1xi32, strided<[2, 1]>>

func.func private @negative_unranked(%arg0: memref<*xi32>) -> memref<*xi32> {
  return %arg0 : memref<*xi32>
}
// CHECK-LABEL:   func.func private @negative_unranked(
// CHECK-SAME:      %[[SRC:.*]]: memref<*xi32>) -> memref<*xi32>
// CHECK:           return %[[SRC]] : memref<*xi32>

func.func @negative_public_function(%arg0: memref<1xi64>) -> memref<1xi64> {
  return %arg0 : memref<1xi64>
}
// CHECK-LABEL:   func.func @negative_public_function(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi64>) -> memref<1xi64>
// CHECK:           return %[[SRC]] : memref<1xi64>

func.func private @negative_callee(%arg0: memref<1xi64>) -> memref<1xi64> {
  return %arg0 : memref<1xi64>
}
// CHECK-LABEL:   func.func private @negative_callee(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi64>) -> memref<1xi64>
// CHECK:           return %[[SRC]] : memref<1xi64>

func.func private @negative_caller(%arg0: memref<1xi64>) -> memref<1xi64> {
  %0 = call @negative_callee(%arg0) : (memref<1xi64>) -> memref<1xi64>
  return %0 : memref<1xi64>
}
// CHECK-LABEL:   func.func private @negative_caller(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi64>) -> memref<1xi64> {
// CHECK:           %[[VAL:.*]] = call @negative_callee(%[[SRC]]) : (memref<1xi64>) -> memref<1xi64>
// CHECK:           return %[[VAL]] : memref<1xi64>
