// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

// CHECK-LABEL: memref_arg
// CHECK-SAME:  !emitc.array<32xf32>)
func.func @memref_arg(%arg0 : memref<32xf32>) {
  func.return
}

// -----

// CHECK-LABEL: memref_return
// CHECK-SAME:  %[[arg0:.*]]: !emitc.array<32xf32>) -> !emitc.array<32xf32>
func.func @memref_return(%arg0 : memref<32xf32>) -> memref<32xf32> {
// CHECK: return %[[arg0]] : !emitc.array<32xf32>
  func.return %arg0 : memref<32xf32>
}

// CHECK-LABEL: memref_call
// CHECK-SAME:  %[[arg0:.*]]: !emitc.array<32xf32>)
func.func @memref_call(%arg0 : memref<32xf32>) {
// CHECK: call @memref_return(%[[arg0]]) : (!emitc.array<32xf32>) -> !emitc.array<32xf32>
  func.call @memref_return(%arg0) : (memref<32xf32>) -> memref<32xf32>
  func.return
}

// -----

// CHECK-LABEL: alloca
func.func @alloca() {
  // CHECK "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>
  return
}

// -----

// CHECK-LABEL: memref_load_store
// CHECK-SAME:  %[[arg0:.*]]: !emitc.array<4x8xf32>, %[[arg1:.*]]: !emitc.array<3x5xf32>
// CHECK-SAME:  %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_load_store(%in: memref<4x8xf32>, %out: memref<3x5xf32>, %i: index, %j: index) {
  // CHECK: %[[load:.*]] = emitc.subscript %[[arg0]][%[[i]], %[[j]]] : <4x8xf32>
  %0 = memref.load %in[%i, %j] : memref<4x8xf32>
  // CHECK: %[[store_loc:.*]] = emitc.subscript %[[arg1]][%[[i]], %[[j]]] : <3x5xf32>
  // CHECK: emitc.assign %[[load]] : f32 to %[[store_loc:.*]] : f32
  memref.store %0, %out[%i, %j] : memref<3x5xf32>
  return
}
