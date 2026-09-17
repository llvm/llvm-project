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

// CHECK-LABEL: func.func @whole_buffer_2d
// CHECK-SAME: (%[[V:.*]]: vector<2x4xf32>)
// CHECK-NOT: memref.alloca
// CHECK-NOT: vector.store
// CHECK-NOT: vector.load
// CHECK: return %[[V]] : vector<2x4xf32>
func.func @whole_buffer_2d(%v: vector<2x4xf32>) -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<2x4xf32>
  vector.store %v, %a[%c0, %c0] : memref<2x4xf32>, vector<2x4xf32>
  %r = vector.load %a[%c0, %c0] : memref<2x4xf32>, vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// The loop carries the current vector value instead of the buffer.
// CHECK-LABEL: func.func @in_loop
// CHECK-SAME: (%[[V:.*]]: vector<4xf32>,
// CHECK-NOT: memref.alloca
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args(%[[IT:.*]] = %[[V]]) -> (vector<4xf32>)
// CHECK-NEXT: %[[NEXT:.*]] = arith.addf %[[IT]], %[[IT]] : vector<4xf32>
// CHECK-NEXT: scf.yield %[[NEXT]] : vector<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[R]] : vector<4xf32>
func.func @in_loop(%v: vector<4xf32>, %lb: index, %ub: index, %step: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<4xf32>
  vector.store %v, %a[%c0] : memref<4xf32>, vector<4xf32>
  scf.for %i = %lb to %ub step %step {
    %old = vector.load %a[%c0] : memref<4xf32>, vector<4xf32>
    %next = arith.addf %old, %old : vector<4xf32>
    vector.store %next, %a[%c0] : memref<4xf32>, vector<4xf32>
  }
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

// Load/store works with the existing subview alias projections.
// CHECK-LABEL: func.func @subview
// CHECK-SAME: (%[[INIT:.*]]: vector<8xf32>, %[[V:.*]]: vector<4xf32>)
// CHECK-NOT: memref.alloca
// CHECK-NOT: memref.subview
// CHECK: %[[INS:.*]] = vector.insert_strided_slice %[[V]], %[[INIT]] {{.*}}offsets = [2], strides = [1]
// CHECK: %[[EXT:.*]] = vector.extract_strided_slice %[[INS]] {{.*}}offsets = [2], sizes = [4], strides = [1]
// CHECK-NEXT: return %[[EXT]] : vector<4xf32>
func.func @subview(%init: vector<8xf32>, %v: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.store %init, %a[%c0] : memref<8xf32>, vector<8xf32>
  %sv = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  vector.store %v, %sv[%c0] : memref<4xf32, strided<[1], offset: 2>>, vector<4xf32>
  %r = vector.load %sv[%c0] : memref<4xf32, strided<[1], offset: 2>>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// The allocator proves that this dynamic buffer has exactly the vector width.
// CHECK-LABEL: func.func @scalable
// CHECK-SAME: (%[[V:.*]]: vector<[4]xf32>)
// CHECK-NOT: memref.alloca
// CHECK-NOT: vector.store
// CHECK-NOT: vector.load
// CHECK: return %[[V]] : vector<[4]xf32>
func.func @scalable(%v: vector<[4]xf32>) -> vector<[4]xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %vs = vector.vscale
  %size = arith.muli %vs, %c4 : index
  %a = memref.alloca(%size) : memref<?xf32>
  vector.store %v, %a[%c0] : memref<?xf32>, vector<[4]xf32>
  %r = vector.load %a[%c0] : memref<?xf32>, vector<[4]xf32>
  return %r : vector<[4]xf32>
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

// -----

// Single-element scalar buffers still use scalar slots, not vector<1x...>.
// CHECK-LABEL: func.func @scalar_slot
// CHECK: memref.alloca
// CHECK: vector.store
// CHECK: vector.load
func.func @scalar_slot(%v: vector<1xf32>) -> vector<1xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<1xf32>
  vector.store %v, %a[%c0] : memref<1xf32>, vector<1xf32>
  %r = vector.load %a[%c0] : memref<1xf32>, vector<1xf32>
  return %r : vector<1xf32>
}

// -----

// A conditional store preserves the previous value on the other path.
// CHECK-LABEL: func.func @conditional_store
// CHECK-SAME: (%[[INIT:.*]]: vector<4xf32>, %[[V:.*]]: vector<4xf32>, %[[COND:.*]]: i1)
// CHECK-NOT: memref.alloca
// CHECK: %[[R:.*]] = scf.if %[[COND]] -> (vector<4xf32>) {
// CHECK-NEXT: scf.yield %[[V]] : vector<4xf32>
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[INIT]] : vector<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[R]] : vector<4xf32>
func.func @conditional_store(%init: vector<4xf32>, %v: vector<4xf32>, %cond: i1) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<4xf32>
  vector.store %init, %a[%c0] : memref<4xf32>, vector<4xf32>
  scf.if %cond {
    vector.store %v, %a[%c0] : memref<4xf32>, vector<4xf32>
  }
  %r = vector.load %a[%c0] : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// An arbitrary dynamic extent cannot be proven to match the vector width.
// CHECK-LABEL: func.func @dynamic_shape
// CHECK: memref.alloca
// CHECK: vector.store
// CHECK: vector.load
func.func @dynamic_shape(%v: vector<4xf32>, %size: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca(%size) : memref<?xf32>
  vector.store %v, %a[%c0] : memref<?xf32>, vector<4xf32>
  %r = vector.load %a[%c0] : memref<?xf32>, vector<4xf32>
  return %r : vector<4xf32>
}
