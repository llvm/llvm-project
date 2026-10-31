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

// A vscale*4-sized allocation matches the whole scalable vector, so plain
// loads and stores can promote it to a single vector SSA value.
// CHECK-LABEL: func.func @scalable_whole_buffer
// CHECK-SAME: (%[[V:.*]]: vector<[4]xf32>)
// CHECK-NOT: memref.alloca
// CHECK-NOT: vector.store
// CHECK-NOT: vector.load
// CHECK: return %[[V]] : vector<[4]xf32>
func.func @scalable_whole_buffer(%v: vector<[4]xf32>) -> vector<[4]xf32> {
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

// -----

// Static subviews still expose their own vector slot. Writes must be inserted
// into the parent value, and reads must extract the same slice.
// CHECK-LABEL: func.func @static_subview
// CHECK-SAME: (%[[V:.*]]: vector<8xf32>, %[[W:.*]]: vector<4xf32>)
// CHECK-NOT: memref.alloca
// CHECK: %[[INSERT:.*]] = vector.insert_strided_slice %[[W]], %[[V]] offsets = [2], strides = [1]
// CHECK: %[[EXTRACT:.*]] = vector.extract_strided_slice %[[INSERT]] offsets = [2], sizes = [4], strides = [1]
// CHECK: return %[[INSERT]], %[[EXTRACT]] : vector<8xf32>, vector<4xf32>
func.func @static_subview(%v: vector<8xf32>, %w: vector<4xf32>) -> (vector<8xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.store %v, %a[%c0] : memref<8xf32>, vector<8xf32>
  %sv = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  vector.store %w, %sv[%c0] : memref<4xf32, strided<[1], offset: 2>>, vector<4xf32>
  %slice = vector.load %sv[%c0] : memref<4xf32, strided<[1], offset: 2>>, vector<4xf32>
  %whole = vector.load %a[%c0] : memref<8xf32>, vector<8xf32>
  return %whole, %slice : vector<8xf32>, vector<4xf32>
}

// -----

// A masked transfer write preserves inactive lanes from the plain store.
// CHECK-LABEL: func.func @mixed_masked_transfer
// CHECK-SAME: (%[[V:.*]]: vector<4xf32>, %[[W:.*]]: vector<4xf32>, %[[M:.*]]: vector<4xi1>)
// CHECK-NOT: memref.alloca
// CHECK: %[[RESULT:.*]] = arith.select %[[M]], %[[W]], %[[V]] : vector<4xi1>, vector<4xf32>
// CHECK-NEXT: return %[[RESULT]] : vector<4xf32>
func.func @mixed_masked_transfer(%v: vector<4xf32>, %w: vector<4xf32>, %mask: vector<4xi1>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<4xf32>
  vector.store %v, %a[%c0] : memref<4xf32>, vector<4xf32>
  vector.transfer_write %w, %a[%c0], %mask {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %r = vector.load %a[%c0] : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A dynamic subview's slot has the parent's vector type, but the load may
// extend beyond the view. Type equality alone must not allow promotion.
// CHECK-LABEL: func.func @negative_dynamic_subview_load
// CHECK: memref.alloca
// CHECK: vector.store
// CHECK: memref.subview
// CHECK: vector.load
func.func @negative_dynamic_subview_load(%v: vector<8xf32>, %n: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.store %v, %a[%c0] : memref<8xf32>, vector<8xf32>
  %sv = memref.subview %a[0] [%n] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  %r = vector.load %sv[%c0] : memref<?xf32, strided<[1]>>, vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// Similarly, a plain store cannot acquire the dynamic alias's masked-write
// semantics merely because its vector type matches the parent slot.
// CHECK-LABEL: func.func @dynamic_subview_store
// CHECK: memref.alloca
// CHECK: vector.store
// CHECK: memref.subview
// CHECK: vector.store
// CHECK: vector.load
func.func @dynamic_subview_store(%v: vector<8xf32>, %w: vector<8xf32>, %n: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.store %v, %a[%c0] : memref<8xf32>, vector<8xf32>
  %sv = memref.subview %a[0] [%n] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  vector.store %w, %sv[%c0] : memref<?xf32, strided<[1]>>, vector<8xf32>
  %r = vector.load %a[%c0] : memref<8xf32>, vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// A plain access to the static parent can still be promoted alongside a
// transfer through a dynamic subview, preserving lanes outside that subview.
// CHECK-LABEL: func.func @mixed_dynamic_subview_transfer
// CHECK-SAME: (%[[V:.*]]: vector<8xf32>, %[[W:.*]]: vector<8xf32>, %[[N:.*]]: index)
// CHECK-NOT: memref.alloca
// CHECK: %[[MASK:.*]] = vector.create_mask %[[N]] : vector<8xi1>
// CHECK: %[[RESULT:.*]] = arith.select %[[MASK]], %[[W]], %[[V]] : vector<8xi1>, vector<8xf32>
// CHECK-NEXT: return %[[RESULT]] : vector<8xf32>
func.func @mixed_dynamic_subview_transfer(%v: vector<8xf32>, %w: vector<8xf32>, %n: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.store %v, %a[%c0] : memref<8xf32>, vector<8xf32>
  %sv = memref.subview %a[0] [%n] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  vector.transfer_write %w, %sv[%c0] {in_bounds = [false]} : vector<8xf32>, memref<?xf32, strided<[1]>>
  %r = vector.load %a[%c0] : memref<8xf32>, vector<8xf32>
  return %r : vector<8xf32>
}
