// RUN: mlir-opt -convert-to-emitc %s | FileCheck %s

/// NOTE: This test intentionally uses `-convert-to-emitc` only.
///
/// The `-convert-memref-to-emitc` pass introduces
/// `builtin.unrealized_conversion_cast` operations when lowering
/// `memref.alloc` results (which are lowered to `emitc.ptr`) to the canonical
/// memref representation used by the type converter (`emitc.array`).
/// These casts are expected at that stage of the pipeline.
///
/// The purpose of this test is to verify the final lowering produced by
/// `-convert-to-emitc`, where `memref.load` and `memref.store` conversions now
/// handle pointer-backed buffers directly and eliminate the intermediate
/// `unrealized_conversion_cast`.
/// Therefore, the test must run the full EmitC conversion pipeline.

/// AllocOp conversion always returns a ptr
// CHECK-LABEL: emitc.func private @memref_alloc_store(
// CHECK-SAME:  %[[VAL:.*]]: f32,
// CHECK-SAME:  %[[ARG_I:.*]]: !emitc.size_t,
// CHECK-SAME:  %[[ARG_J:.*]]: !emitc.size_t)
func.func private @memref_alloc_store(%v : f32, %i: index, %j: index) {
  /// Allocation size  computation
  // CHECK:     %[[SIZEOF_F32:.*]] = call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t
  // CHECK:     %[[NUM_ELEMS:.*]] = "emitc.constant"() <{value = 32 : index}> : () -> index
  // CHECK:     %[[TOTAL_BYTES:.*]] = mul %[[SIZEOF_F32]], %[[NUM_ELEMS]] : (!emitc.size_t, index) -> !emitc.size_t
  /// Alloc
  // CHECK:     %[[MALLOC_PTR:.*]] = call_opaque "malloc"(%[[TOTAL_BYTES]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK:     %[[ELEM_PTR:.*]] = cast %[[MALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<f32>
  %alloc = memref.alloc() : memref<4x8xf32>
  /// Store subscript computation
  // CHECK:     %[[ROW_STRIDE:.*]] = "emitc.constant"() <{value = 8 : index}> : () -> !emitc.size_t
  // CHECK:     %[[ROW_OFFSET:.*]] = mul %[[ARG_I]], %[[ROW_STRIDE]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK:     %[[LINEAR_INDEX:.*]] = add %[[ROW_OFFSET]], %[[ARG_J]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK:     %[[ELEM_LVALUE:.*]] = subscript %[[ELEM_PTR]]{{\[}}%[[LINEAR_INDEX]]] : (!emitc.ptr<f32>, !emitc.size_t) -> !emitc.lvalue<f32>
  /// Store
  // CHECK:     assign %[[VAL]] : f32 to %[[ELEM_LVALUE]] : <f32>
  memref.store %v, %alloc[%i, %j] : memref<4x8xf32>
  return
}
// CHECK-LABEL: emitc.func private @memref_alloc_load(
// CHECK-SAME:  %[[ARG_I:.*]]: !emitc.size_t,
// CHECK-SAME:  %[[ARG_J:.*]]: !emitc.size_t) -> f32
func.func private @memref_alloc_load(%i: index, %j: index) -> f32 {
  // CHECK:     %[[SIZEOF_F32:.*]] = call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t
  // CHECK:     %[[NUM_ELEMS:.*]] = "emitc.constant"() <{value = 32 : index}> : () -> index
  // CHECK:     %[[TOTAL_BYTES:.*]] = mul %[[SIZEOF_F32]], %[[NUM_ELEMS]] : (!emitc.size_t, index) -> !emitc.size_t
  // CHECK:     %[[MALLOC_PTR:.*]] = call_opaque "malloc"(%[[TOTAL_BYTES]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK:     %[[ELEM_PTR:.*]] = cast %[[MALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<f32>
  %alloc = memref.alloc() : memref<4x8xf32>
  // CHECK:     %[[ROW_STRIDE:.*]] = "emitc.constant"() <{value = 8 : index}> : () -> !emitc.size_t
  // CHECK:     %[[ROW_OFFSET:.*]] = mul %[[ARG_I]], %[[ROW_STRIDE]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK:     %[[LINEAR_INDEX:.*]] = add %[[ROW_OFFSET]], %[[ARG_J]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK:     %[[ELEM_LVALUE:.*]] = subscript %[[ELEM_PTR]]{{\[}}%[[LINEAR_INDEX]]] : (!emitc.ptr<f32>, !emitc.size_t) -> !emitc.lvalue<f32>
  // CHECK:     %[[LOADED_VAL:.*]] = load %[[ELEM_LVALUE]] : <f32>
  %v = memref.load %alloc[%i, %j] : memref<4x8xf32>
  return %v : f32
}

/// LoadOp and StoreOp still compatible
/// Previous array paths still available
// CHECK-LABEL: emitc.func @memref_load_store(
// CHECK-SAME:  %[[BUFF0:.*]]: !emitc.array<2xf32>,
// CHECK-SAME:  %[[BUFF1:.*]]: !emitc.array<4x8xf32>,
func.func @memref_load_store(%buff0: memref<2xf32>,
  %buff1: memref<4x8xf32>, %i : index, %j : index) {
  // CHECK:     %[[ELEM_LVALUE0:.*]] = subscript %[[BUFF0]]{{.*}} (!emitc.array<2xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  // CHECK:     %[[VAL:.*]] = load %[[ELEM_LVALUE0]] : <f32>
  %v = memref.load %buff0[%i] : memref<2xf32>
  // CHECK:     %[[ELEM_LVALUE1:.*]] = subscript %[[BUFF1]]{{.*}} (!emitc.array<4x8xf32>, !emitc.size_t, !emitc.size_t) -> !emitc.lvalue<f32>
  // CHECK:     assign %[[VAL]] : f32 to %[[ELEM_LVALUE1]] : <f32>
  memref.store %v, %buff1[%i, %j] : memref<4x8xf32>
  return
}
