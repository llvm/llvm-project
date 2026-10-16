// RUN: mlir-opt %s -mem2reg -split-input-file | FileCheck %s

// Reading a static buffer through a dynamic subview returns the stored value
// within the subview's extent and the transfer's padding outside it.

// CHECK-LABEL: func.func @read_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         %[[MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x16xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[V]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @read_dyn_subview(%v: vector<8x16xf32>, %n: index, %pad: f32)
    -> vector<8x16xf32> {
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

// Write through a dynamic subview, then read the whole buffer. The write updates
// only the elements within the dynamic extent, so the read returns the written
// value for those and the buffer's earlier value for the rest.

// CHECK-LABEL: func.func @write_then_read_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[W:.*]]: vector<8x16xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[W]], %[[V]]
// CHECK:         return %[[SEL]]
func.func @write_then_read_dyn_subview(%v: vector<8x16xf32>, %w: vector<8x16xf32>,
                                       %n: index, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]}
      : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1]
      : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  vector.transfer_write %w, %sv[%c0, %c0] {in_bounds = [true, false]}
      : vector<8x16xf32>, memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]}
      : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A read with a mask through a dynamic subview combines both masks: the subview
// extent (vector.create_mask) and the transfer's own mask operand.
// CHECK-LABEL: func.func @read_with_mask_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[M:.*]]: vector<8x16xi1>, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK:         %[[CM:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[AND:.*]] = arith.andi %[[CM]], %[[M]]
// CHECK:         %[[SEL:.*]] = arith.select %[[AND]], %[[V]], %{{.*}}
// CHECK:         return %[[SEL]]
func.func @read_with_mask_dyn_subview(%v: vector<8x16xf32>, %n: index, %m: vector<8x16xi1>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad, %m {in_bounds = [true, false]} : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A dynamic subview of a static subview uses the static subview's shape (4x16)
// for its mask.

// CHECK-LABEL: func.func @read_dyn_subview_of_static_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview
// CHECK:         %[[EX:.*]] = vector.extract_strided_slice %[[V]] offsets = [2, 0], sizes = [4, 16]
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<4x16xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast
// CHECK:         %[[SEL:.*]] = arith.select %[[CM]], %[[EX]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @read_dyn_subview_of_static_subview(%v: vector<8x16xf32>, %n: index,
                                         %pad: f32) -> vector<4x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv1 = memref.subview %a[2, 0] [4, 16] [1, 1] : memref<8x16xf32> to memref<4x16xf32, strided<[16, 1], offset: 32>>
  %sv2 = memref.subview %sv1[0, 0] [%n, 16] [1, 1] : memref<4x16xf32, strided<[16, 1], offset: 32>> to memref<?x16xf32, strided<[16, 1], offset: 32>>
  %r = vector.transfer_read %sv2[%c0, %c0], %pad {in_bounds = [false, true]} : memref<?x16xf32, strided<[16, 1], offset: 32>>, vector<4x16xf32>
  return %r : vector<4x16xf32>
}

// -----

// The promoted value flows across control flow: the `scf.if` returns the
// masked store on one path and the untouched value on the other.

// CHECK-LABEL: func.func @write_dyn_subview_in_scf_if(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %[[W:[a-z0-9_]+]]: vector<8x16xf32>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[IF:.*]] = scf.if %{{.*}} -> (vector<8x16xf32>) {
// CHECK:           %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<8x16xi1>
// CHECK:           %[[SEL:.*]] = arith.select %[[CM]], %[[W]], %[[V]]
// CHECK:           scf.yield %[[SEL]] : vector<8x16xf32>
// CHECK:         } else {
// CHECK:           scf.yield %[[V]] : vector<8x16xf32>
// CHECK:         }
// CHECK:         return %[[IF]]
func.func @write_dyn_subview_in_scf_if(%v: vector<8x16xf32>, %w: vector<8x16xf32>,
                                 %n: index, %pad: f32, %c: i1) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  scf.if %c {
    %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
    vector.transfer_write %w, %sv[%c0, %c0] {in_bounds = [false, true]} : vector<8x16xf32>, memref<?x16xf32, strided<[16, 1]>>
  }
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
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
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]}
      : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, %off] [8, %n] [1, 1]
      : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1], offset: ?>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [true, false]}
      : memref<8x?xf32, strided<[16, 1], offset: ?>>, vector<8x16xf32>
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
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[2, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1], offset: 32>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [false, true]} : memref<?x16xf32, strided<[16, 1], offset: 32>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Collapse a dynamic subview by reshaping both the parent vector and the
// subview mask. Lanes outside the runtime bounds receive the read's padding.
// CHECK-LABEL: func.func @collapse_shape_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.collapse_shape
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[V]] : vector<8x16xf32> to vector<128xf32>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<8x16xi1>
// CHECK:         %[[MASK:.*]] = vector.shape_cast %[[CM]] : vector<8x16xi1> to vector<128xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<128xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[SC]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @collapse_shape_dyn_subview(%v: vector<8x16xf32>, %n: index, %pad: f32) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %flat = memref.collapse_shape %sv [[0, 1]] : memref<?x16xf32, strided<[16, 1]>> into memref<?xf32, strided<[1]>>
  %r = vector.transfer_read %flat[%c0], %pad {in_bounds = [false]} : memref<?xf32, strided<[1]>>, vector<128xf32>
  return %r : vector<128xf32>
}


// -----

// Expanding the dynamic subview gives a static alias shape of 8x2, derived
// from the parent vector's 16 elements. The memref extent remains dynamic;
// the reshaped subview mask determines which lanes are valid.
// CHECK-LABEL: func.func @expand_shape_dyn_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<16xf32>, %[[M:[a-z0-9_]+]]: index,
// CHECK-SAME:      %{{[a-z0-9_]+}}: index, %[[PAD:[a-z0-9_]+]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.expand_shape
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[V]] : vector<16xf32> to vector<8x2xf32>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[M]] : vector<16xi1>
// CHECK:         %[[MASK:.*]] = vector.shape_cast %[[CM]] : vector<16xi1> to vector<8x2xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x2xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[SC]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @expand_shape_dyn_subview(%v: vector<16xf32>, %m: index, %d: index,
                                    %pad: f32) -> vector<8x2xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<16xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
  %sv = memref.subview %a[0] [%m] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [%d, 2] : memref<?xf32, strided<[1]>> into memref<?x2xf32>
  %r = vector.transfer_read %exp[%c0, %c0], %pad {in_bounds = [false, true]} : memref<?x2xf32>, vector<8x2xf32>
  return %r : vector<8x2xf32>
}


// -----

// A chain of reshapes casts the value at each step (8x16 -> 128 -> 64x2).
// The read builds the subview mask in the parent shape and casts it directly
// to the final shape.

// CHECK-LABEL: func.func @collapse_then_expand_dyn_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.collapse_shape
// CHECK-NOT:     memref.expand_shape
// CHECK:         %[[F:.*]] = vector.shape_cast %[[V]] : vector<8x16xf32> to vector<128xf32>
// CHECK:         %[[E:.*]] = vector.shape_cast %[[F]] : vector<128xf32> to vector<64x2xf32>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<8x16xi1>
// CHECK:         %[[MASK:.*]] = vector.shape_cast %[[CM]] : vector<8x16xi1> to vector<64x2xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[E]], %{{.*}}
// CHECK:         return %[[SEL]]
func.func @collapse_then_expand_dyn_subview(%v: vector<8x16xf32>, %n: index,
                                            %d: index, %pad: f32) -> vector<64x2xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %f = memref.collapse_shape %sv [[0, 1]] : memref<?x16xf32, strided<[16, 1]>> into memref<?xf32, strided<[1]>>
  %e = memref.expand_shape %f [[0, 1]] output_shape [%d, 2] : memref<?xf32, strided<[1]>> into memref<?x2xf32>
  %r = vector.transfer_read %e[%c0, %c0], %pad {in_bounds = [false, true]} : memref<?x2xf32>, vector<64x2xf32>
  return %r : vector<64x2xf32>
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
  %a = memref.alloca() : memref<16xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
  %sv = memref.subview %a[0] [%m] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [%d0, %d1] : memref<?xf32, strided<[1]>> into memref<?x?xf32>
  %r = vector.transfer_read %exp[%c0, %c0], %pad : memref<?x?xf32>, vector<8x2xf32>
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
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[0] [%n] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [2, %m] : memref<?xf32, strided<[1]>> into memref<2x?xf32>
  %r = vector.transfer_read %exp[%c0, %c0], %pad {in_bounds = [true, false]} : memref<2x?xf32>, vector<2x4xf32>
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
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %e = memref.expand_shape %sv [[0, 1], [2]] output_shape [%d, 0, 16] : memref<?x16xf32, strided<[16, 1]>> into memref<?x0x16xf32, strided<[0, 16, 1]>>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad : memref<?x0x16xf32, strided<[0, 16, 1]>>, vector<8x16xf32>
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
  %a = memref.alloca() : memref<4x4xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32>
  %sv = memref.subview %a[0, 0] [4, %n] [1, 1] : memref<4x4xf32> to memref<4x?xf32, strided<[4, 1]>>
  %f = memref.collapse_shape %sv [[0, 1]] : memref<4x?xf32, strided<[4, 1]>> into memref<?xf32, strided<[?]>>
  %r = vector.transfer_read %f[%c0], %pad {in_bounds = [false]} : memref<?xf32, strided<[?]>>, vector<16xf32>
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
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 8] [1, 1] : memref<8x16xf32> to memref<?x8xf32, strided<[16, 1]>>
  %exp = memref.expand_shape %sv [[0, 1], [2]] output_shape [%d, 2, 8] : memref<?x8xf32, strided<[16, 1]>> into memref<?x2x8xf32, strided<[32, 16, 1]>>
  %r = vector.transfer_read %exp[%c0, %c0, %c0], %pad {in_bounds = [false, true, true]} : memref<?x2x8xf32, strided<[32, 16, 1]>>, vector<8x2x8xf32>
  return %r : vector<8x2x8xf32>
}
