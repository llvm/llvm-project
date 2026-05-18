// RUN: mlir-opt -convert-to-emitc -reconcile-unrealized-casts %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive tests
//===----------------------------------------------------------------------===//

func.func private @rank0_alloc() -> memref<i32> {
  %alloc = memref.alloc() : memref<i32>
  return %alloc : memref<i32>
}
// CHECK-LABEL:   emitc.func private @rank0_alloc() -> i32
// CHECK:           %[[SIZEOF_I32:.*]] = call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK:           %[[NUM_ELEMS:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
// CHECK:           %[[TOTAL_BYTES:.*]] = mul %[[SIZEOF_I32]], %[[NUM_ELEMS]] : (!emitc.size_t, index) -> !emitc.size_t
// CHECK:           %[[MALLOC_PTR:.*]] = call_opaque "malloc"(%[[TOTAL_BYTES]])
// CHECK-SAME:        : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK:           %[[ELEM_PTR:.*]] = cast %[[MALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK:           %[[INDEX:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[ELEM_LVALUE:.*]] = subscript %[[ELEM_PTR]][%[[INDEX]]]
// CHECK-SAME:        : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
// CHECK:           %[[LOADED_VAL:.*]] = load %[[ELEM_LVALUE]] : <i32>
// CHECK:           return %[[LOADED_VAL]] : i32
// CHECK:         }

func.func private @rank0_arg(%arg0: memref<i64>) -> memref<i64> {
  return %arg0 : memref<i64>
}
// CHECK-LABEL:   emitc.func private @rank0_arg(
// CHECK-SAME:      %[[SRC:.*]]: !emitc.ptr<i64>) -> i64
// CHECK:           %[[INDEX:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[ELEM_LVALUE:.*]] = subscript %[[SRC]][%[[INDEX]]]
// CHECK-SAME:        : (!emitc.ptr<i64>, index) -> !emitc.lvalue<i64>
// CHECK:           %[[LOADED_VAL:.*]] = load %[[ELEM_LVALUE]] : <i64>
// CHECK:           return %[[LOADED_VAL]] : i64
// CHECK:         }

memref.global "private" constant @c : memref<i64> = dense<-1>
func.func private @rank0_constant() -> memref<i64> {
  %0 = memref.get_global @c : memref<i64>
  return %0 : memref<i64>
}
// CHECK:         emitc.global static const @c : i64 = -1
// CHECK-LABEL:   emitc.func private @rank0_constant() -> i64
// CHECK:           %[[CONST:.*]] = get_global @c : !emitc.lvalue<i64>
// CHECK:           %[[ADDR:.*]] = address_of %[[CONST]] : !emitc.lvalue<i64>
// CHECK:           %[[INDEX:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[ELEM_LVALUE:.*]] = subscript %[[ADDR]][%[[INDEX]]]
// CHECK-SAME:        : (!emitc.ptr<i64>, index) -> !emitc.lvalue<i64>
// CHECK:           %[[LOADED_VAL:.*]] = load %[[ELEM_LVALUE]] : <i64>
// CHECK:           return %[[LOADED_VAL]] : i64
// CHECK:         }

func.func private @rank1_alloc() -> memref<1xi32> {
  %alloc = memref.alloc() : memref<1xi32>
  return %alloc : memref<1xi32>
}
// CHECK-LABEL:   emitc.func private @rank1_alloc() -> i32
// CHECK:           %[[SIZEOF_I32:.*]] = call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK:           %[[NUM_ELEMS:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> index
// CHECK:           %[[TOTAL_BYTES:.*]] = mul %[[SIZEOF_I32]], %[[NUM_ELEMS]] : (!emitc.size_t, index) -> !emitc.size_t
// CHECK:           %[[MALLOC_PTR:.*]] = call_opaque "malloc"(%[[TOTAL_BYTES]])
// CHECK-SAME:        : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK:           %[[ELEM_PTR:.*]] = cast %[[MALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK:           %[[INDEX:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[ELEM_LVALUE:.*]] = subscript %[[ELEM_PTR]][%[[INDEX]]]
// CHECK-SAME:        : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
// CHECK:           %[[LOADED_VAL:.*]] = load %[[ELEM_LVALUE]] : <i32>
// CHECK:           return %[[LOADED_VAL]] : i32
// CHECK:         }

func.func private @rank1_arg(%arg0: memref<1xi64>) -> memref<1xi64> {
  return %arg0 : memref<1xi64>
}
// CHECK-LABEL:   emitc.func private @rank1_arg(
// CHECK-SAME:      %[[SRC:.*]]: !emitc.array<1xi64>) -> i64
// CHECK:           %[[C0:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[ELEM_LVALUE:.*]] = subscript %[[SRC]][%[[C0]]] :
// CHECK-SAME:        (!emitc.array<1xi64>, index) -> !emitc.lvalue<i64>
// CHECK:           %[[LOADED_VAL:.*]] = load %[[ELEM_LVALUE]] : <i64>
// CHECK:           return %[[LOADED_VAL]] : i64
// CHECK:         }

func.func private @rank2_arg(%arg0: memref<1x1xi64>) -> memref<1x1xi64> {
  return %arg0 : memref<1x1xi64>
}
// CHECK-LABEL:   emitc.func private @rank2_arg(
// CHECK-SAME:      %[[SRC:.*]]: !emitc.array<1x1xi64>) -> i64
// CHECK:           %[[C0:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK:           %[[ELEM_LVALUE:.*]] = subscript %[[SRC]][%[[C0]], %[[C0]]] :
// CHECK-SAME:        (!emitc.array<1x1xi64>, index, index) -> !emitc.lvalue<i64>
// CHECK:           %[[LOADED_VAL:.*]] = load %[[ELEM_LVALUE]] : <i64>
// CHECK:           return %[[LOADED_VAL]] : i64
// CHECK:         }

//===----------------------------------------------------------------------===//
// Negative tests (must NOT rewrite)
//===----------------------------------------------------------------------===//

func.func private @negative_multiple_returns(%arg0: memref<1xi64>) -> (memref<1xi64>, i64) {
  %1 = arith.constant 7 : i64
  return %arg0, %1 : memref<1xi64>, i64
}
// CHECK-LABEL:   func.func private @negative_multiple_returns(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi64>) -> (memref<1xi64>, i64)
// CHECK:           %[[C0:.*]] = "emitc.constant"() <{value = 7 : i64}> : () -> i64
// CHECK:           return %[[SRC]], %[[C0]] : memref<1xi64>, i64
// CHECK:         }

func.func private @negative_dynamic_shape(%arg0: memref<?xi64>) -> memref<?xi64> {
  return %arg0 : memref<?xi64>
}
// CHECK-LABEL:   func.func private @negative_dynamic_shape(
// CHECK-SAME:      %[[SRC:.*]]: memref<?xi64>) -> memref<?xi64>
// CHECK:           return %[[SRC]] : memref<?xi64>
// CHECK:         }

func.func private @negative_non_identity_layout(%arg0: memref<1x1xi64, strided<[2, 1], offset: 0>>)
    -> memref<1x1xi64, strided<[2, 1], offset: 0>> {
  return %arg0 : memref<1x1xi64, strided<[2, 1], offset: 0>>
}
// CHECK-LABEL:   func.func private @negative_non_identity_layout(
// CHECK-SAME:      %[[SRC:.*]]: memref<1x1xi64, strided<[2, 1]>>) -> memref<1x1xi64, strided<[2, 1]>>
// CHECK:           return %[[SRC]] : memref<1x1xi64, strided<[2, 1]>>
// CHECK:         }

func.func private @negative_unranked(%arg0: memref<*xi64>) -> memref<*xi64> {
  return %arg0 : memref<*xi64>
}
// CHECK-LABEL:   func.func private @negative_unranked(
// CHECK-SAME:      %[[SRC:.*]]: memref<*xi64>) -> memref<*xi64>
// CHECK:           return %[[SRC]] : memref<*xi64>
// CHECK:         }
