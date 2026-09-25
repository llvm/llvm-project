// RUN: mlir-opt %s --mem2reg --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @whole_buffer
// CHECK-SAME: (%[[V:.*]]: vector<4xf32>)
// CHECK-NOT: memref.alloca
// CHECK-NOT: vector.store
// CHECK-NOT: vector.load
// CHECK: return %[[V]] : vector<4xf32>
func.func @whole_buffer(%v: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<4xf32>
  vector.store %v, %a[%c0] : memref<4xf32>, vector<4xf32>
  %r = vector.load %a[%c0] : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// Both kinds of vector access can use the same promoted slot.
// CHECK-LABEL: func.func @mixed_transfers
// CHECK-SAME: (%[[V:.*]]: vector<4xf32>,
// CHECK-NOT: memref.alloca
// CHECK-NOT: vector.transfer
// CHECK-NOT: vector.store
// CHECK-NOT: vector.load
// CHECK: return %[[V]] : vector<4xf32>
func.func @mixed_transfers(%v: vector<4xf32>, %pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<4xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %loaded = vector.load %a[%c0] : memref<4xf32>, vector<4xf32>
  vector.store %loaded, %a[%c0] : memref<4xf32>, vector<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A partial read blocks promotion even when the store covers the whole buffer.
// CHECK-LABEL: func.func @partial_load
// CHECK: memref.alloca
// CHECK: vector.store
// CHECK: vector.load
func.func @partial_load(%v: vector<8xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.store %v, %a[%c0] : memref<8xf32>, vector<8xf32>
  %r = vector.load %a[%c0] : memref<8xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A partial store also blocks promotion.
// CHECK-LABEL: func.func @partial_store
// CHECK: memref.alloca
// CHECK: vector.store
// CHECK: vector.load
func.func @partial_store(%v: vector<4xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.store %v, %a[%c0] : memref<8xf32>, vector<4xf32>
  %r = vector.load %a[%c0] : memref<8xf32>, vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// Exact vector type alone is insufficient: the access must start at zero.
// CHECK-LABEL: func.func @nonzero_index
// CHECK: memref.alloca
// CHECK: vector.store
// CHECK: vector.load
func.func @nonzero_index(%v: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a = memref.alloca() : memref<4xf32>
  vector.store %v, %a[%c0] : memref<4xf32>, vector<4xf32>
  %r = vector.load %a[%c1] : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A dynamic store index cannot be proven to cover the whole buffer.
// CHECK-LABEL: func.func @dynamic_index
// CHECK: memref.alloca
// CHECK: vector.store
// CHECK: vector.load
func.func @dynamic_index(%v: vector<4xf32>, %i: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<4xf32>
  vector.store %v, %a[%i] : memref<4xf32>, vector<4xf32>
  %r = vector.load %a[%c0] : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}
