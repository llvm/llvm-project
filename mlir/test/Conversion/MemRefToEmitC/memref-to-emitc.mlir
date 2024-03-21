// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

// CHECK-LABEL: memref_store
// CHECK-SAME:  %[[v:.*]]: f32, %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_store(%v : f32, %i: index, %j: index) {
  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>

  // CHECK: %[[SUBSCRIPT:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : <4x8xf32>
  // CHECK: emitc.assign %[[v]] : f32 to %[[SUBSCRIPT:.*]] : f32
  memref.store %v, %0[%i, %j] : memref<4x8xf32>
  return
}
// -----

// CHECK-LABEL: memref_load
// CHECK-SAME:  %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_load(%i: index, %j: index) -> f32 {
  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>

  // CHECK: %[[LOAD:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : <4x8xf32>
  // CHECK: %[[VAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  // CHECK: emitc.assign %[[LOAD]] : f32 to %[[VAR]] : f32
  %1 = memref.load %0[%i, %j] : memref<4x8xf32>
  // CHECK: return %[[VAR]] : f32
  return %1 : f32
}
