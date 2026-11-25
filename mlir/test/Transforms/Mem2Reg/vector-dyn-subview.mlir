// RUN: mlir-opt %s -mem2reg -split-input-file | FileCheck %s

// Write through a dynamic subview, then read the whole buffer. The write is
// masked to the dynamic extent, so the read returns the written value for those
// elements and the buffer's earlier value for the rest.

// CHECK-LABEL: func.func @write_then_read_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[W:.*]]: vector<8x16xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     vector.mask
// CHECK:         %[[M:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[STORED:.*]] = arith.select %[[M]], %[[W]], %[[V]]
// CHECK:         %[[CM:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[CM]], %[[STORED]], %[[V]]
// CHECK:         return %[[SEL]]
func.func @write_then_read_dyn_subview(%v: vector<8x16xf32>, %w: vector<8x16xf32>,
                                       %n: index, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]}
      : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1]
      : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %m = vector.create_mask %c8, %n : vector<8x16xi1>
  vector.mask %m {
    vector.transfer_write %w, %sv[%c0, %c0] {in_bounds = [true, true]}
        : vector<8x16xf32>, memref<8x?xf32, strided<[16, 1]>>
  } : vector<8x16xi1>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]}
      : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A read through a dynamic subview with a mask operand: disabled lanes use
// padding. The operand mask is intersected with the runtime extent mask.
// CHECK-LABEL: func.func @read_with_mask_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[M:.*]]: vector<8x16xi1>, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview
// CHECK:         %[[EXTENT:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[MASK:.*]] = arith.andi %[[M]], %[[EXTENT]] : vector<8x16xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x16xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[V]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @read_with_mask_dyn_subview(%v: vector<8x16xf32>, %n: index, %m: vector<8x16xi1>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad, %m : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}
// -----

// A read wrapped in `vector.mask`: the mask (here covering the subview's
// extent) disables lanes past the extent, which use padding. The subview adds
// no extent mask.
// CHECK-LABEL: func.func @masked_read_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     vector.mask
// CHECK:         %[[M:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK-NOT:     vector.create_mask
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x16xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[V]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @masked_read_dyn_subview(%v: vector<8x16xf32>, %n: index, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %m = vector.create_mask %c8, %n : vector<8x16xi1>
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.mask %m { vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Lanes disabled by vector.mask use the passthru; enabled out-of-bounds
// lanes use the transfer padding.
// CHECK-LABEL: func.func @masked_read_passthru_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[M:.*]]: vector<8x16xi1>, %[[PT:.*]]: vector<8x16xf32>, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     vector.mask
// CHECK:         %[[EXTENT:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x16xf32>
// CHECK:         %[[READ:.*]] = arith.select %[[EXTENT]], %[[V]], %[[PS]]
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[READ]], %[[PT]]
// CHECK:         return %[[SEL]]
func.func @masked_read_passthru_dyn_subview(%v: vector<8x16xf32>, %n: index, %m: vector<8x16xi1>, %pt: vector<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.mask %m, %pt { vector.transfer_read %sv[%c0, %c0], %pad : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Update only write lanes enabled by the mask and inside the dynamic subview.
// Preserve the parent's previous values elsewhere.
// CHECK-LABEL: func.func @masked_write_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[W:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[M:.*]]: vector<8x16xi1>
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     vector.mask
// CHECK:         %[[STORED:.*]] = arith.select %[[M]], %[[W]], %[[V]]
// CHECK:         %[[CM:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[CM]], %[[STORED]], %[[V]]
// CHECK:         return %[[SEL]]
func.func @masked_write_dyn_subview(%v: vector<8x16xf32>, %w: vector<8x16xf32>, %n: index, %m: vector<8x16xi1>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  vector.mask %m { vector.transfer_write %w, %sv[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x?xf32, strided<[16, 1]>> } : vector<8x16xi1>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}


// -----

// A dynamic subview of a static subview aliases the static subview's whole
// value (4x16). The read is masked to the dynamic extent.

// CHECK-LABEL: func.func @read_dyn_subview_of_static_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %[[N:[a-z0-9_]+]]: index, %[[PAD:[a-z0-9_]+]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview
// CHECK-NOT:     vector.mask
// CHECK:         %[[M:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<4x16xi1>
// CHECK:         %[[EX:.*]] = vector.extract_strided_slice %[[V]] offsets = [2, 0], sizes = [4, 16]
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<4x16xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[EX]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @read_dyn_subview_of_static_subview(%v: vector<8x16xf32>, %n: index,
                                         %pad: f32) -> vector<4x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv1 = memref.subview %a[2, 0] [4, 16] [1, 1] : memref<8x16xf32> to memref<4x16xf32, strided<[16, 1], offset: 32>>
  %sv2 = memref.subview %sv1[0, 0] [%n, 16] [1, 1] : memref<4x16xf32, strided<[16, 1], offset: 32>> to memref<?x16xf32, strided<[16, 1], offset: 32>>
  %m = vector.create_mask %n, %c16 : vector<4x16xi1>
  %r = vector.mask %m { vector.transfer_read %sv2[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x16xf32, strided<[16, 1], offset: 32>>, vector<4x16xf32> } : vector<4x16xi1> -> vector<4x16xf32>
  return %r : vector<4x16xf32>
}

// -----

// The promoted value flows across control flow: the `scf.if` returns the
// masked store on one path and the untouched value on the other.

// CHECK-LABEL: func.func @write_dyn_subview_in_scf_if(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %[[W:[a-z0-9_]+]]: vector<8x16xf32>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     vector.mask
// CHECK:         %[[IF:.*]] = scf.if %{{.*}} -> (vector<8x16xf32>) {
// CHECK:           %[[M:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<8x16xi1>
// CHECK:           %[[STORED:.*]] = arith.select %[[M]], %[[W]], %[[V]]
// CHECK:           %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<8x16xi1>
// CHECK:           %[[SEL:.*]] = arith.select %[[CM]], %[[STORED]], %[[V]]
// CHECK:           scf.yield %[[SEL]] : vector<8x16xf32>
// CHECK:         } else {
// CHECK:           scf.yield %[[V]] : vector<8x16xf32>
// CHECK:         }
// CHECK:         return %[[IF]]
func.func @write_dyn_subview_in_scf_if(%v: vector<8x16xf32>, %w: vector<8x16xf32>,
                                 %n: index, %pad: f32, %c: i1) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  scf.if %c {
    %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
    %m = vector.create_mask %n, %c16 : vector<8x16xi1>
    vector.mask %m { vector.transfer_write %w, %sv[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<?x16xf32, strided<[16, 1]>> } : vector<8x16xi1>
  }
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A dynamic subview's slot value represents its parent, not its runtime shape.
// Reads past the runtime extent use padding, including when size < offset.
// CHECK-LABEL: func.func @subview_dynamic_partial_read
// CHECK-SAME: (%[[V:.*]]: vector<8xf32>, %[[PAD:.*]]: f32, %[[SIZE:.*]]: index)
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK-NOT: memref.alloca
// CHECK-NOT: memref.subview
// CHECK: %[[SLICE:.*]] = vector.extract_strided_slice %[[V]] offsets = [2], sizes = [4], strides = [1]
// CHECK: %[[BOUND:.*]] = arith.subi %[[SIZE]], %[[C2]] : index
// CHECK: %[[MASK:.*]] = vector.create_mask %[[BOUND]] : vector<4xi1>
// CHECK: %[[PADDING:.*]] = vector.broadcast %[[PAD]] : f32 to vector<4xf32>
// CHECK: %[[RESULT:.*]] = arith.select %[[MASK]], %[[SLICE]], %[[PADDING]] : vector<4xi1>, vector<4xf32>
// CHECK: return %[[RESULT]]
func.func @subview_dynamic_partial_read(%v: vector<8xf32>, %pad: f32, %size: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %v, %a[%c0] : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[0] [%size] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  %r = vector.transfer_read %sv[%c2], %pad : memref<?xf32, strided<[1]>>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// Whole-buffer reads also pad lanes outside the dynamic extent.
// CHECK-LABEL: func.func @read_dyn_subview_out_of_bounds(
// CHECK-SAME: %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[PAD:.*]]: f32
// CHECK-NOT: memref.alloca
// CHECK-NOT: memref.subview
// CHECK: %[[MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK: %[[PADDING:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x16xf32>
// CHECK: %[[RESULT:.*]] = arith.select %[[MASK]], %[[V]], %[[PADDING]]
// CHECK: return %[[RESULT]]
func.func @read_dyn_subview_out_of_bounds(%v: vector<8x16xf32>, %n: index,
                                         %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0]
      : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1]
      : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [true, false]}
      : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Negative: a subview with a dynamic offset prevents buffer promotion.

// CHECK-LABEL: func.func @negative_read_dyn_subview_dynamic_offset(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         vector.transfer_read
func.func @negative_read_dyn_subview_dynamic_offset(%v: vector<8x16xf32>, %off: index, %n: index,
                              %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]}
      : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, %off] [8, %n] [1, 1]
      : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1], offset: ?>>
  %m = vector.create_mask %c8, %n : vector<8x16xi1>
  %r = vector.mask %m {
    vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [true, true]}
        : memref<8x?xf32, strided<[16, 1], offset: ?>>, vector<8x16xf32>
  } : vector<8x16xi1> -> vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Negative: a dynamic size with a static but non-zero offset. The valid region
// no longer starts at the parent's origin, which `vector.create_mask` cannot
// express, so the buffer is not promoted.
// CHECK-LABEL: func.func @negative_read_dyn_subview_static_offset_dyn_size(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         vector.transfer_read
func.func @negative_read_dyn_subview_static_offset_dyn_size(%v: vector<8x16xf32>, %n: index,
                                      %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[2, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1], offset: 32>>
  %m = vector.create_mask %n, %c16 : vector<8x16xi1>
  %r = vector.mask %m { vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x16xf32, strided<[16, 1], offset: 32>>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
  return %r : vector<8x16xf32>
}
