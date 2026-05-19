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
  /// The returned SSA value is produced by memref.get_global. The memref.global
  /// above only declares the storage symbol and cannot be returned directly.
  /// Both of these are supported in EmitC lowering.
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
  // Even though memref<1xi32> normally converts to an EmitC array type, memref
  // allocations are heap-backed and lower to a pointer. The scalarized return
  // therefore uses the same pointer load path as rank-0 memrefs.
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

func.func private @negative_rank0_constant() -> memref<i64> {
  // Arith memref constants are not storage-producing operations. Lowering this
  // through EmitC would require turning a dense memref attribute into pointer
  // or array storage, so the Func conversion must leave this function unchanged.
  %0 = arith.constant dense<-1> : memref<i64>
  return %0 : memref<i64>
}
// CHECK-LABEL:   func.func private @negative_rank0_constant() -> memref<i64> {
// CHECK:           %[[CONST:.*]] = arith.constant dense<-1> : memref<i64>
// CHECK:           return %[[CONST]] : memref<i64>
// CHECK:         }

func.func private @negative_rank1_constant() -> memref<1xi64> {
  // Same as the rank-0 case: the memref-typed constant is not a supported
  // returned storage producer even though the result has one element.
  %0 = arith.constant dense<[-1]> : memref<1xi64>
  return %0 : memref<1xi64>
}
// CHECK-LABEL:   func.func private @negative_rank1_constant() -> memref<1xi64> {
// CHECK:           %[[CONST:.*]] = arith.constant dense<-1> : memref<1xi64>
// CHECK:           return %[[CONST]] : memref<1xi64>
// CHECK:         }

func.func private @negative_multiple_elements(%arg0: memref<2xi64>) -> memref<2xi64> {
  return %arg0 : memref<2xi64>
}
// CHECK-LABEL:   func.func private @negative_multiple_elements(
// CHECK-SAME:      %[[SRC:.*]]: memref<2xi64>) -> memref<2xi64>
// CHECK:           return %[[SRC]] : memref<2xi64>
// CHECK:         }

func.func private @negative_multiple_returns(%arg0: memref<1xi32>) -> (memref<1xi32>, i32) {
  %1 = arith.constant 7 : i32
  return %arg0, %1 : memref<1xi32>, i32
}
// CHECK-LABEL:   func.func private @negative_multiple_returns(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi32>) -> (memref<1xi32>, i32)
// CHECK:           %[[C0:.*]] = "emitc.constant"() <{value = 7 : i32}> : () -> i32
// CHECK:           return %[[SRC]], %[[C0]] : memref<1xi32>, i32
// CHECK:         }

func.func private @negative_dynamic_shape(%arg0: memref<?xi32>) -> memref<?xi32> {
  return %arg0 : memref<?xi32>
}
// CHECK-LABEL:   func.func private @negative_dynamic_shape(
// CHECK-SAME:      %[[SRC:.*]]: memref<?xi32>) -> memref<?xi32>
// CHECK:           return %[[SRC]] : memref<?xi32>
// CHECK:         }

func.func private @negative_non_identity_layout(%arg0: memref<1x1xi32, strided<[2, 1], offset: 0>>)
    -> memref<1x1xi32, strided<[2, 1], offset: 0>> {
  return %arg0 : memref<1x1xi32, strided<[2, 1], offset: 0>>
}
// CHECK-LABEL:   func.func private @negative_non_identity_layout(
// CHECK-SAME:      %[[SRC:.*]]: memref<1x1xi32, strided<[2, 1]>>) -> memref<1x1xi32, strided<[2, 1]>>
// CHECK:           return %[[SRC]] : memref<1x1xi32, strided<[2, 1]>>
// CHECK:         }

func.func private @negative_unranked(%arg0: memref<*xi32>) -> memref<*xi32> {
  return %arg0 : memref<*xi32>
}
// CHECK-LABEL:   func.func private @negative_unranked(
// CHECK-SAME:      %[[SRC:.*]]: memref<*xi32>) -> memref<*xi32>
// CHECK:           return %[[SRC]] : memref<*xi32>
// CHECK:         }

func.func @negative_public_function(%arg0: memref<1xi64>) -> memref<1xi64> {
  // Public functions keep their API. Scalarizing this result would change the
  // externally visible signature from memref<1xi64> to i64.
  return %arg0 : memref<1xi64>
}
// CHECK-LABEL:   func.func @negative_public_function(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi64>) -> memref<1xi64>
// CHECK:           return %[[SRC]] : memref<1xi64>
// CHECK:         }

func.func private @negative_callee(%arg0: memref<1xi64>) -> memref<1xi64> {
  // This function is private but it has a symbol user below. It must not be
  // scalarized unless all call sites are updated as part of a broader API
  // rewrite.
  return %arg0 : memref<1xi64>
}
// CHECK-LABEL:   func.func private @negative_callee(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi64>) -> memref<1xi64>
// CHECK:           return %[[SRC]] : memref<1xi64>
// CHECK:         }

func.func private @negative_caller(%arg0: memref<1xi64>) -> memref<1xi64> {
  // Returning the result of a memref-returning call is intentionally rejected:
  // the narrow scalarization only handles values already backed by supported
  // storage producers such as block arguments, memref.alloc, and get_global.
  %0 = call @negative_callee(%arg0) : (memref<1xi64>) -> memref<1xi64>
  return %0 : memref<1xi64>
}
// CHECK-LABEL:   func.func private @negative_caller(
// CHECK-SAME:      %[[SRC:.*]]: memref<1xi64>) -> memref<1xi64> {
// CHECK:           %[[VAL:.*]] = call @negative_callee(%[[SRC]]) : (memref<1xi64>) -> memref<1xi64>
// CHECK:           return %[[VAL]] : memref<1xi64>
// CHECK:         }
