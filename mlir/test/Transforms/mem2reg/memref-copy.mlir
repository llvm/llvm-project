// RUN: mlir-opt %s -mem2reg -split-input-file | FileCheck %s


// `memref.copy` participates in promotion as a transfer of the buffer's value.
// Copying into the whole slot and reading it becomes a read of the source.

// CHECK-LABEL: func.func @copy_in_whole(
// CHECK-SAME:      %[[SRC:.*]]: memref<8x16xf32>, %[[PAD:.*]]: f32
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.copy
//        CHECK:   %[[R:.*]] = vector.transfer_read %[[SRC]]{{.*}} : memref<8x16xf32>, vector<8x16xf32>
//        CHECK:   return %[[R]]
func.func @copy_in_whole(%src: memref<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  memref.copy %src, %a : memref<8x16xf32> to memref<8x16xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Copying out of the slot becomes a transfer_write of its value to the target.

// CHECK-LABEL: func.func @copy_out(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[DST:.*]]: memref<8x16xf32>
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.copy
//        CHECK:   vector.transfer_write %[[V]], %[[DST]]
func.func @copy_out(%v: vector<8x16xf32>, %dst: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %a, %dst : memref<8x16xf32> to memref<8x16xf32>
  return
}

// -----

// Copying between two promotable slots eliminates both buffers.

// CHECK-LABEL: func.func @copy_between_slots(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.copy
//        CHECK:   return %[[V]]
func.func @copy_between_slots(%v: vector<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
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

// CHECK-LABEL: func.func @negative_self_copy(
//        CHECK:   memref.alloca
//        CHECK:   memref.copy
//        CHECK:   vector.transfer_read
func.func @negative_self_copy(%v: vector<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
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

// CHECK-LABEL: func.func @negative_scalable_copy(
//        CHECK:   memref.alloca
//        CHECK:   memref.copy
//        CHECK:   vector.transfer_read
func.func @negative_scalable_copy(%src: memref<8xf32>, %pad: f32) -> vector<[4]xf32> {
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
// becomes a read of the source, kept only where it falls inside the extent.

// CHECK-LABEL: func.func @copy_into_dyn_subview(
// CHECK-SAME:      %[[SRC:.*]]: memref<?x?xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[Z:.*]] = vector.broadcast %{{.*}} : f32 to vector<8x16xf32>
// CHECK:         %[[RD:.*]] = vector.transfer_read %[[SRC]]{{.*}} : memref<?x?xf32>, vector<8x16xf32>
// CHECK:         %[[M:.*]] = vector.create_mask %[[N]], %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[RD]], %[[Z]]
// CHECK:         return %[[SEL]]
func.func @copy_into_dyn_subview(%src: memref<?x?xf32>, %n: index, %pad: f32) -> vector<8x16xf32> {
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

// CHECK-LABEL: func.func @copy_dynsub_to_dynsub(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[SRC:.*]]: memref<8x16xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[SS:.*]] = memref.subview %[[SRC]][0, 0] [8, %[[N]]] [1, 1]
// CHECK:         %[[RD:.*]] = vector.transfer_read %[[SS]]{{.*}} {in_bounds = [true, false]}
// CHECK:         %[[M:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[RD]], %[[V]]
// CHECK:         return %[[SEL]]
func.func @copy_dynsub_to_dynsub(%v: vector<8x16xf32>, %src: memref<8x16xf32>, %n: index, %pad: f32) -> vector<8x16xf32> {
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

// Copy from a global into a dynamic subview of a static staging buffer, in a
// non-default memory space.
memref.global "private" constant @tile : memref<128x64xf16> = dense<1.000000e+00>

// CHECK-LABEL: func.func @copy_global_to_dyn_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<128x64xf16>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[G:.*]] = memref.get_global @tile : memref<128x64xf16>
// CHECK:         %[[GSV:.*]] = memref.subview %[[G]][0, 0] [%[[N]], 64] [1, 1]
// CHECK:         %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:         %[[RD:.*]] = vector.transfer_read %[[GSV]][%{{.*}}, %{{.*}}], %[[PAD]] {in_bounds = [false, true]} : memref<?x64xf16, strided<[64, 1]>>, vector<128x64xf16>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<128x64xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[CM]], %[[RD]], %[[V]]
// CHECK:         return %[[SEL]]
func.func @copy_global_to_dyn_subview(%v: vector<128x64xf16>, %n: index, %pad: f16) -> vector<128x64xf16> {
  %c0 = arith.constant 0 : index
  %g = memref.get_global @tile : memref<128x64xf16>
  %gsv = memref.subview %g[0, 0] [%n, 64] [1, 1] : memref<128x64xf16> to memref<?x64xf16, strided<[64, 1]>>
  %slm = memref.alloca() : memref<128x64xf16, 3>
  vector.transfer_write %v, %slm[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf16>, memref<128x64xf16, 3>
  %sv = memref.subview %slm[0, 0] [%n, 64] [1, 1] : memref<128x64xf16, 3> to memref<?x64xf16, strided<[64, 1]>, 3>
  memref.copy %gsv, %sv : memref<?x64xf16, strided<[64, 1]>> to memref<?x64xf16, strided<[64, 1]>, 3>
  %r = vector.transfer_read %slm[%c0, %c0], %pad {in_bounds = [true, true]} : memref<128x64xf16, 3>, vector<128x64xf16>
  return %r : vector<128x64xf16>
}

// -----

// A copy out of a dynamic subview becomes an unmasked transfer_write: the
// target has the subview's runtime shape, so the write's out-of-bounds lanes
// fall outside it.

// CHECK-LABEL: func.func @copy_out_of_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[DST:.*]]: memref<?x?xf32>
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK-NOT:     vector.create_mask
// CHECK:         vector.transfer_write %[[V]], %[[DST]][%{{.*}}, %{{.*}}] : vector<8x16xf32>, memref<?x?xf32>
// CHECK-NEXT:    return
func.func @copy_out_of_dyn_subview(%v: vector<8x16xf32>, %dst: memref<?x?xf32>, %n: index) {
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

// CHECK-LABEL: func.func @copy_dyn_subview_between_slots(
// CHECK-SAME:      %[[VA:.*]]: vector<8x16xf32>, %[[VB:.*]]: vector<8x16xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[M:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[VA]], %[[VB]]
// CHECK:         return %[[SEL]]
func.func @copy_dyn_subview_between_slots(%va: vector<8x16xf32>, %vb: vector<8x16xf32>, %n: index, %pad: f32) -> vector<8x16xf32> {
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
// promoted.

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

// NEGATIVE: a copy between two static subviews of the same buffer, shifting
// the low half into the high half, is not promoted.

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
