// RUN: mlir-opt %s -mem2reg -split-input-file | FileCheck %s


// `memref.copy` participates in promotion as a transfer of the buffer's value.
// Copying into the whole slot and reading it becomes a read of the source.

// CHECK-LABEL: func.func @copy_src(
// CHECK-SAME:      %[[SRC:.*]]: memref<8x16xf32>, %[[PAD:.*]]: f32
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.copy
//        CHECK:   %[[R:.*]] = vector.transfer_read %[[SRC]]{{.*}} : memref<8x16xf32>, vector<8x16xf32>
//        CHECK:   return %[[R]]
func.func @copy_src(%src: memref<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  memref.copy %src, %a : memref<8x16xf32> to memref<8x16xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Copying out of the slot becomes a transfer_write of its value to the target.

// CHECK-LABEL: func.func @copy_into_dst(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[DST:.*]]: memref<8x16xf32>
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.copy
//        CHECK:   vector.transfer_write %[[V]], %[[DST]]
func.func @copy_into_dst(%v: vector<8x16xf32>, %dst: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %a, %dst : memref<8x16xf32> to memref<8x16xf32>
  return
}

// -----

// Copying between two promotable slots eliminates both buffers.

// CHECK-LABEL: func.func @copy_intermediate(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.copy
//        CHECK:   return %[[V]]
func.func @copy_intermediate(%v: vector<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  %b = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %a, %b : memref<8x16xf32> to memref<8x16xf32>
  %r = vector.transfer_read %b[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// NEGATIVE: a self-copy references the slot on both sides and is not modeled, so
// the buffer is not promoted.

// CHECK-LABEL: func.func @negative_copy_self(
//        CHECK:   memref.alloca
//        CHECK:   memref.copy
//        CHECK:   vector.transfer_read
func.func @negative_copy_self(%v: vector<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %a, %a : memref<8x16xf32> to memref<8x16xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// NEGATIVE: copies of scalable slots are not supported. Such buffers can still
// be promoted through transfers (see @scalable_whole_buffer_in_loop).

// CHECK-LABEL: func.func @negative_copy_src_scalable(
//        CHECK:   memref.alloca
//        CHECK:   memref.copy
//        CHECK:   vector.transfer_read
func.func @negative_copy_src_scalable(%src: memref<8xf32>, %pad: f32) -> vector<[4]xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %vs = vector.vscale
  %sz = arith.muli %vs, %c4 : index
  %a = memref.alloca(%sz) : memref<?xf32>
  memref.copy %src, %a : memref<8xf32> to memref<?xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<?xf32>, vector<[4]xf32>
  return %r : vector<[4]xf32>
}

// -----

// A `memref.copy` into a dynamic subview, read back afterwards: the copy
// becomes a read of the source masked to the subview's extent, kept only
// where it falls inside the extent.

// CHECK-LABEL: func.func @copy_src_into_dyn_subview(
// CHECK-SAME:      %[[SRC:.*]]: memref<?x?xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[Z:.*]] = vector.broadcast %{{.*}} : f32 to vector<8x16xf32>
// CHECK:         %[[READ_MASK:.*]] = vector.create_mask %[[N]], %[[N]] : vector<8x16xi1>
// CHECK:         %[[RD:.*]] = vector.transfer_read %[[SRC]]{{.*}}, %[[READ_MASK]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<8x16xf32>
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[N]], %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[RD]], %[[Z]]
// CHECK:         return %[[SEL]]
func.func @copy_src_into_dyn_subview(%src: memref<?x?xf32>, %n: index, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %z = arith.constant 0.000000e+00 : f32
  %zv = vector.broadcast %z : f32 to vector<8x16xf32>
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %zv, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, %n] [1, 1] : memref<8x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %src, %sv : memref<?x?xf32> to memref<?x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Copy from one dynamic subview into another: the source subview is read, and
// only the elements inside the extent reach the buffer's value.

// CHECK-LABEL: func.func @copy_dyn_subview_to_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[SRC:.*]]: memref<8x16xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[SS:.*]] = memref.subview %[[SRC]][0, 0] [8, %[[N]]] [1, 1]
// CHECK:         %[[READ_MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[RD:.*]] = vector.transfer_read %[[SS]]{{.*}}, %[[READ_MASK]] {in_bounds = [true, true]}
// CHECK:         %[[MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[RD]], %[[V]]
// CHECK:         return %[[SEL]]
func.func @copy_dyn_subview_to_dyn_subview(%v: vector<8x16xf32>, %src: memref<8x16xf32>, %n: index, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %ssrc = memref.subview %src[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %sdst = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  memref.copy %ssrc, %sdst : memref<8x?xf32, strided<[16, 1]>> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// The source of the copy can be a global.
memref.global "private" constant @src : memref<8x16xf32> = dense<1.000000e+00>

// CHECK-LABEL: func.func @copy_src_global(
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.copy
//        CHECK:   %[[G:.*]] = memref.get_global @src : memref<8x16xf32>
//        CHECK:   %[[R:.*]] = vector.transfer_read %[[G]]{{.*}} : memref<8x16xf32>, vector<8x16xf32>
//        CHECK:   return %[[R]]
func.func @copy_src_global(%pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %g = memref.get_global @src : memref<8x16xf32>
  %a = memref.alloca() : memref<8x16xf32>
  memref.copy %g, %a : memref<8x16xf32> to memref<8x16xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A copy out of a dynamic subview becomes a transfer_write of the slot's value,
// masked to the subview's extent.

// CHECK-LABEL: func.func @copy_dyn_subview_into_dst(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[DST:.*]]: memref<?x?xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[N]], %[[N]] : vector<8x16xi1>
// CHECK:         vector.transfer_write %[[V]], %[[DST]][%{{.*}}, %{{.*}}], %[[MASK]] {in_bounds = [true, true]} : vector<8x16xf32>, memref<?x?xf32>
// CHECK-NEXT:    return
func.func @copy_dyn_subview_into_dst(%v: vector<8x16xf32>, %dst: memref<?x?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, %n] [1, 1] : memref<8x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %sv, %dst : memref<?x?xf32, strided<[16, 1]>> to memref<?x?xf32>
  return
}

// -----

// Copying between dynamic subviews of two promotable slots eliminates both
// buffers: only the elements inside the extent reach the target's value.

// CHECK-LABEL: func.func @copy_intermediate_dyn_subview(
// CHECK-SAME:      %[[VA:.*]]: vector<8x16xf32>, %[[VB:.*]]: vector<8x16xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[WRITE_MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[W:.*]] = arith.select %[[WRITE_MASK]], %[[VA]], %[[VB]]
// CHECK:         %[[MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[W]], %[[VB]]
// CHECK:         return %[[SEL]]
func.func @copy_intermediate_dyn_subview(%va: vector<8x16xf32>, %vb: vector<8x16xf32>, %n: index, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  %b = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %va, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  vector.transfer_write %vb, %b[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sa = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %sb = memref.subview %b[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  memref.copy %sa, %sb : memref<8x?xf32, strided<[16, 1]>> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %b[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// NEGATIVE: a copy between two dynamic subviews of the same buffer is not
// promoted: Mem2Reg rejects a memory op that reaches one slot through two
// distinct aliases, as `PromotableMemOpInterface` handles a single slot per
// call.

// CHECK-LABEL: func.func @negative_copy_dyn_subviews_same_buffer(
// CHECK:         memref.alloca
// CHECK:         memref.copy
// CHECK:         vector.transfer_read
func.func @negative_copy_dyn_subviews_same_buffer(%v: vector<8x16xf32>, %n: index, %m: index, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %s1 = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %s2 = memref.subview %a[0, 0] [8, %m] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  memref.copy %s1, %s2 : memref<8x?xf32, strided<[16, 1]>> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// NEGATIVE: as above, with two static subviews of the same buffer, shifting
// the low half into the high half.

// CHECK-LABEL: func.func @negative_copy_static_subviews_same_buffer(
// CHECK:         memref.alloca
// CHECK:         memref.copy
// CHECK:         vector.transfer_read
func.func @negative_copy_static_subviews_same_buffer(%v: vector<8xf32>, %pad: f32) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %lo = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  %hi = memref.subview %a[4] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 4>>
  memref.copy %lo, %hi : memref<4xf32, strided<[1]>> to memref<4xf32, strided<[1], offset: 4>>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  return %r : vector<8xf32>
}
