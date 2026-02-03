// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s
// RUN: mlir-opt -convert-to-emitc="filter-dialects=memref" %s -split-input-file | FileCheck %s

// CHECK-LABEL: alloca()
func.func @alloca() {
  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<2xf32>
  // CHECK-NEXT: [[CONST:%.+]] = "emitc.constant"() <{value = 0 : index}> : () -> index
  // CHECK-NEXT: [[SUBSCRIPT:%.+]] = emitc.subscript %[[ALLOCA]]{{.}}[[CONST]]{{.}} : (!emitc.array<2xf32>, index) -> !emitc.lvalue<f32>
  // CHECK-NEXT: [[ADDR_OF:%.+]] = emitc.address_of [[SUBSCRIPT]] : !emitc.lvalue<f32>
  // CHECK-NEXT: [[CAST:%.+]] = emitc.cast [[ADDR_OF]] : !emitc.ptr<f32> to !emitc.ptr<!emitc.array<2xf32>>
  %0 = memref.alloca() : memref<2xf32>
  return
}

// -----
// CHECK:         emitc.verbatim "\0A/* Generalized Indexing Template */
// CHECK-SAME:\0Atemplate <typename T> constexpr T mt_index(T i_last)
// CHECK-SAME: { return i_last; }\0Atemplate <typename T, typename... Args>
// CHECK-SAME: \0Aconstexpr T mt_index(T idx, T stride, Args... rest)
// CHECK-SAME: {\0A    return (idx * stride) + mt_index(rest...);\0A}\0A"

// CHECK-LABEL: memref_store
// CHECK-SAME:   ([[BUFF:%.+]]: memref<4x8xf32>, [[PARAM_1_:%.+]]: f32, [[INDEX_1_:%.+]]: index, [[INDEX_2_:%.+]]: index) {
func.func @memref_store(%buff : memref<4x8xf32>, %v : f32, %i: index, %j: index) {
// CHECK-DAG:       [[BUFFER:%.+]] = builtin.unrealized_conversion_cast [[BUFF]] : memref<4x8xf32> to !emitc.ptr<!emitc.array<4x8xf32>>
// CHECK-DAG:       [[CONST_1:%.+]] = "emitc.constant"() <{value = 8 : index}> : () -> index
// CHECK-DAG:       [[CALL_OPAQUE:%.+]] = emitc.call_opaque "mt_index"([[INDEX_1_]], [[CONST_1]], [[INDEX_2_]]) : (index, index, index) -> index
// CHECK-DAG:       [[CAST:%.+]] = emitc.cast [[BUFFER]] : !emitc.ptr<!emitc.array<4x8xf32>> to !emitc.ptr<f32>
// CHECK:           [[SUBSCRIPT:%.+]] = emitc.subscript [[CAST]]{{.}}[[CALL_OPAQUE]]{{.}} : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
// CHECK:           emitc.assign [[PARAM_1_]] : f32 to [[SUBSCRIPT]] : <f32>

  memref.store %v, %buff[%i, %j] : memref<4x8xf32>
  return
}

// -----

// CHECK-LABEL: memref_load
// CHECK-SAME:  %[[buff:.*]]: memref<4x8xf32>, %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_load(%buff : memref<4x8xf32>, %i: index, %j: index) -> f32 {
// CHECK-DAG:       [[BUFFER:%.+]] = builtin.unrealized_conversion_cast %[[buff]] : memref<4x8xf32> to !emitc.ptr<!emitc.array<4x8xf32>>
// CHECK-DAG:       [[CONST_1:%.+]] = "emitc.constant"() <{value = 8 : index}> : () -> index
// CHECK-DAG:       [[CALL_OPAQUE:%.+]] = emitc.call_opaque "mt_index"(%[[i]], [[CONST_1]], %[[j]]) : (index, index, index) -> index
// CHECK-DAG:       [[CAST:%.+]] = emitc.cast [[BUFFER]] : !emitc.ptr<!emitc.array<4x8xf32>> to !emitc.ptr<f32>
// CHECK:           [[SUBSCRIPT:%.+]] = emitc.subscript [[CAST]]{{.}}[[CALL_OPAQUE]]{{.}} : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
// CHECK:           [[LOAD_SUBSCRIPT:%.+]] = emitc.load [[SUBSCRIPT]] : <f32>
  %1 = memref.load %buff[%i, %j] : memref<4x8xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: globals
module @globals {
  memref.global "private" constant @internal_global : memref<3x7xf32> = dense<4.0>
  // CHECK-NEXT: emitc.global static const @internal_global : !emitc.array<3x7xf32> = dense<4.000000e+00>
  memref.global "private" constant @__constant_xi32 : memref<i32> = dense<-1>
  // CHECK-NEXT: emitc.global static const @__constant_xi32 : i32 = -1
  memref.global @public_global : memref<3x7xf32>
  // CHECK-NEXT: emitc.global extern @public_global : !emitc.array<3x7xf32>
  memref.global @uninitialized_global : memref<3x7xf32> = uninitialized
  // CHECK-NEXT: emitc.global extern @uninitialized_global : !emitc.array<3x7xf32>

  // CHECK-LABEL: use_global
  func.func @use_global() {
    // CHECK-NEXT: emitc.get_global @public_global : !emitc.array<3x7xf32>
    // CHECK-NEXT: "emitc.constant"() <{value = 0 : index}> : () -> index
    // CHECK-NEXT: "emitc.constant"() <{value = 0 : index}> : () -> index
    // CHECK: emitc.subscript %0[%1, %2] : (!emitc.array<3x7xf32>, index, index) -> !emitc.lvalue<f32>
    // CHECK-NEXT: emitc.address_of %3 : !emitc.lvalue<f32>
    // CHECK-NEXT: emitc.cast %4 : !emitc.ptr<f32> to !emitc.ptr<!emitc.array<3x7xf32>>
    %0 = memref.get_global @public_global : memref<3x7xf32>
    // CHECK-NEXT: emitc.get_global @__constant_xi32 : !emitc.lvalue<i32>
    // CHECK-NEXT: emitc.address_of %6 : !emitc.lvalue<i32>
    // CHECK-NEXT: emitc.cast %7 : !emitc.ptr<i32> to !emitc.ptr<!emitc.array<1xi32>>
    %1 = memref.get_global @__constant_xi32 : memref<i32>
    return
  }
}
