// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

// CHECK-LABEL: memref
// CHECK-SAME:  %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref(%i: index, %j: index) {
  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>

  // CHECK: %[[LOAD:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : <4x8xf32>
  %1 = memref.load %0[%i, %j] : memref<4x8xf32>

  // CHECK: %[[SUBSCRIPT:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : <4x8xf32>
  // CHECK: emitc.assign %[[LOAD]] : f32 to %[[SUBSCRIPT:.*]] : f32
  memref.store %1, %0[%i, %j] : memref<4x8xf32>
  return
}
