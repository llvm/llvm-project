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
// padding. The subview adds no extent mask.
// CHECK-LABEL: func.func @read_with_mask_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[M:.*]]: vector<8x16xi1>, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     vector.create_mask
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x16xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[V]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @read_with_mask_dyn_subview(%v: vector<8x16xf32>, %n: index, %m: vector<8x16xi1>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad, %m {in_bounds = [true, true]} : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32>
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

// Lanes disabled by `vector.mask` use the passthru, not the padding.
// CHECK-LABEL: func.func @masked_read_passthru_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[PT:.*]]: vector<8x16xf32>, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     vector.mask
// CHECK:         %[[M:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK-NOT:     vector.create_mask
// CHECK-NOT:     vector.broadcast
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[V]], %[[PT]]
// CHECK:         return %[[SEL]]
func.func @masked_read_passthru_dyn_subview(%v: vector<8x16xf32>, %n: index, %pt: vector<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %m = vector.create_mask %c8, %n : vector<8x16xi1>
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.mask %m, %pt { vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
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

// Collapse a dynamic subview by reshaping both the parent vector and the
// subview mask. The read's mask (here covering the collapsed extent) disables
// lanes past the runtime bounds, which use padding. The subview adds no extent
// mask.
// CHECK-LABEL: func.func @collapse_shape_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.collapse_shape
// CHECK-NOT:     vector.mask
// CHECK:         %[[LEN:.*]] = arith.muli %[[N]], %{{.*}} : index
// CHECK:         %[[M:.*]] = vector.create_mask %[[LEN]] : vector<128xi1>
// CHECK-NOT:     vector.create_mask
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[V]] : vector<8x16xf32> to vector<128xf32>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<128xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[SC]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @collapse_shape_dyn_subview(%v: vector<8x16xf32>, %n: index, %pad: f32) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %len = arith.muli %n, %c16 : index
  %m = vector.create_mask %len : vector<128xi1>
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %flat = memref.collapse_shape %sv [[0, 1]] : memref<?x16xf32, strided<[16, 1]>> into memref<?xf32, strided<[1]>>
  %r = vector.mask %m { vector.transfer_read %flat[%c0], %pad {in_bounds = [true]} : memref<?xf32, strided<[1]>>, vector<128xf32> } : vector<128xi1> -> vector<128xf32>
  return %r : vector<128xf32>
}

// -----

// Expanding the dynamic subview gives a static alias shape of 8x2, derived
// from the parent vector's 16 elements. The memref extent remains dynamic;
// the read's mask determines which lanes are valid. The subview adds no
// extent mask.
// CHECK-LABEL: func.func @expand_shape_dyn_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<16xf32>, %{{[a-z0-9_]+}}: index,
// CHECK-SAME:      %[[D:[a-z0-9_]+]]: index, %[[PAD:[a-z0-9_]+]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.expand_shape
// CHECK-NOT:     vector.mask
// CHECK:         %[[M:.*]] = vector.create_mask %[[D]], %{{.*}} : vector<8x2xi1>
// CHECK-NOT:     vector.create_mask
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[V]] : vector<16xf32> to vector<8x2xf32>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x2xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[SC]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @expand_shape_dyn_subview(%v: vector<16xf32>, %m: index, %d: index,
                                    %pad: f32) -> vector<8x2xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %mask = vector.create_mask %d, %c2 : vector<8x2xi1>
  %a = memref.alloca() : memref<16xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
  %sv = memref.subview %a[0] [%m] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [%d, 2] : memref<?xf32, strided<[1]>> into memref<?x2xf32>
  %r = vector.mask %mask { vector.transfer_read %exp[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x2xf32>, vector<8x2xf32> } : vector<8x2xi1> -> vector<8x2xf32>
  return %r : vector<8x2xf32>
}


// -----

// A chain of reshapes casts the value at each step (8x16 -> 128 -> 64x2).
// The read's mask, given in the final shape, selects the valid lanes; the
// subview adds no extent mask.

// CHECK-LABEL: func.func @collapse_then_expand_dyn_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %{{[a-z0-9_]+}}: index,
// CHECK-SAME:      %[[D:[a-z0-9_]+]]: index, %[[PAD:[a-z0-9_]+]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.collapse_shape
// CHECK-NOT:     memref.expand_shape
// CHECK-NOT:     vector.mask
// CHECK:         %[[M:.*]] = vector.create_mask %[[D]], %{{.*}} : vector<64x2xi1>
// CHECK-NOT:     vector.create_mask
// CHECK:         %[[F:.*]] = vector.shape_cast %[[V]] : vector<8x16xf32> to vector<128xf32>
// CHECK:         %[[E:.*]] = vector.shape_cast %[[F]] : vector<128xf32> to vector<64x2xf32>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<64x2xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[E]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @collapse_then_expand_dyn_subview(%v: vector<8x16xf32>, %n: index,
                                            %d: index, %pad: f32) -> vector<64x2xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %m = vector.create_mask %d, %c2 : vector<64x2xi1>
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %f = memref.collapse_shape %sv [[0, 1]] : memref<?x16xf32, strided<[16, 1]>> into memref<?xf32, strided<[1]>>
  %e = memref.expand_shape %f [[0, 1]] output_shape [%d, 2] : memref<?xf32, strided<[1]>> into memref<?x2xf32>
  %r = vector.mask %m { vector.transfer_read %e[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x2xf32>, vector<64x2xf32> } : vector<64x2xi1> -> vector<64x2xf32>
  return %r : vector<64x2xf32>
}

// -----

// Negative: an out-of-bounds transfer (`in_bounds = [false]`) through a dynamic
// subview is not promoted; lanes past the extent must be masked off instead.
// TODO: Remove once out-of-bounds transfers are removed (#227821).

// CHECK-LABEL: func.func @negative_read_dyn_subview_out_of_bounds(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         vector.transfer_read
func.func @negative_read_dyn_subview_out_of_bounds(%v: vector<8x16xf32>, %n: index,
                                                   %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]}
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

// -----

// Reject two dynamic dimensions in one group: the parent extent alone cannot
// determine their individual extents or the resulting row stride.
// CHECK-LABEL: func.func @neg_two_dynamic_dims_in_group(
// CHECK:         memref.alloca
// CHECK:         memref.expand_shape
// CHECK:         vector.transfer_read
func.func @neg_two_dynamic_dims_in_group(%v: vector<16xf32>, %m: index, %d0: index,
                                         %d1: index, %pad: f32) -> vector<8x2xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %d0, %d1 : vector<8x2xi1>
  %a = memref.alloca() : memref<16xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
  %sv = memref.subview %a[0] [%m] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [%d0, %d1] : memref<?xf32, strided<[1]>> into memref<?x?xf32>
  %r = vector.mask %mask { vector.transfer_read %exp[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xf32>, vector<8x2xf32> } : vector<8x2xi1> -> vector<8x2xf32>
  return %r : vector<8x2xf32>
}

// -----

// Reject expansion to 2x?: the memref's row stride is %m, whereas a 2x4 vector
// has row stride 4. A mask cannot correct this difference in element positions.
// CHECK-LABEL: func.func @neg_nonunit_dim_before_dynamic(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         memref.expand_shape
// CHECK:         vector.transfer_read
func.func @neg_nonunit_dim_before_dynamic(%v: vector<8xf32>, %n: index, %m: index,
                                          %pad: f32) -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %mask = vector.create_mask %c2, %m : vector<2x4xi1>
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[0] [%n] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [2, %m] : memref<?xf32, strided<[1]>> into memref<2x?xf32>
  %r = vector.mask %mask { vector.transfer_read %exp[%c0, %c0], %pad {in_bounds = [true, true]} : memref<2x?xf32>, vector<2x4xf32> } : vector<2x4xi1> -> vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// Reject a zero static extent without dividing by zero during shape inference.
// CHECK-LABEL: func.func @neg_zero_extent_in_group(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         memref.expand_shape
// CHECK:         vector.transfer_read
func.func @neg_zero_extent_in_group(%v: vector<8x16xf32>, %n: index, %d: index,
                                    %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %m = vector.create_mask %c0, %c16 : vector<8x16xi1>
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %e = memref.expand_shape %sv [[0, 1], [2]] output_shape [%d, 0, 16] : memref<?x16xf32, strided<[16, 1]>> into memref<?x0x16xf32, strided<[0, 16, 1]>>
  %r = vector.mask %m { vector.transfer_read %e[%c0, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x0x16xf32, strided<[0, 16, 1]>>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Reject collapsing 4x?: the subview's rows have stride 4, so they are not
// guaranteed contiguous when the inner extent is %n.
// CHECK-LABEL: func.func @neg_collapse_dyn_inner_dim(
// CHECK:         memref.alloca
// CHECK:         memref.collapse_shape
// CHECK:         vector.transfer_read
func.func @neg_collapse_dyn_inner_dim(%v: vector<4x4xf32>, %n: index,
                                      %pad: f32) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %len = arith.muli %n, %c4 : index
  %m = vector.create_mask %len : vector<16xi1>
  %a = memref.alloca() : memref<4x4xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32>
  %sv = memref.subview %a[0, 0] [4, %n] [1, 1] : memref<4x4xf32> to memref<4x?xf32, strided<[4, 1]>>
  %f = memref.collapse_shape %sv [[0, 1]] : memref<4x?xf32, strided<[4, 1]>> into memref<?xf32, strided<[?]>>
  %r = vector.mask %m { vector.transfer_read %f[%c0], %pad {in_bounds = [true]} : memref<?xf32, strided<[?]>>, vector<16xf32> } : vector<16xi1> -> vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// Reject an expansion whose fully static group accounts for only 8 of the
// parent dimension's 16 elements. The inferred alias shape would lose half
// the parent vector's elements, violating the shape_cast element-count rule.
// CHECK-LABEL: func.func @neg_static_group_narrowed_subview(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         memref.expand_shape
// CHECK:         vector.transfer_read
func.func @neg_static_group_narrowed_subview(%v: vector<8x16xf32>, %n: index,
                                             %d: index, %pad: f32) -> vector<8x2x8xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %m = vector.create_mask %d, %c2, %c8 : vector<8x2x8xi1>
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 8] [1, 1] : memref<8x16xf32> to memref<?x8xf32, strided<[16, 1]>>
  %exp = memref.expand_shape %sv [[0, 1], [2]] output_shape [%d, 2, 8] : memref<?x8xf32, strided<[16, 1]>> into memref<?x2x8xf32, strided<[32, 16, 1]>>
  %r = vector.mask %m { vector.transfer_read %exp[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]} : memref<?x2x8xf32, strided<[32, 16, 1]>>, vector<8x2x8xf32> } : vector<8x2x8xi1> -> vector<8x2x8xf32>
  return %r : vector<8x2x8xf32>
}
