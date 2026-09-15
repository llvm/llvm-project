// RUN: mlir-opt %s -mem2reg -canonicalize -split-input-file | FileCheck %s

// memref.copy participates in Mem2Reg as a vector transfer of the buffer.

// Copy INTO the whole slot then read -> read of the source; buffer eliminated.
// CHECK-LABEL: func.func @copy_in_whole(
// CHECK-SAME:      %[[SRC:.*]]: memref<8x16xf32>, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         %[[R:.*]] = vector.transfer_read %[[SRC]]{{.*}} : memref<8x16xf32>, vector<8x16xf32>
// CHECK:         return %[[R]]
func.func @copy_in_whole(%src: memref<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  memref.copy %src, %a : memref<8x16xf32> to memref<8x16xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A copy into a dynamic subview whose value IS read is preserved as a masked
// select composing the source over the prior buffer value.
// CHECK-LABEL: func.func @copy_live(
// CHECK-SAME:      %[[SRC:.*]]: memref<?x?xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK-DAG:     %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK:         %[[RD:.*]] = vector.transfer_read %[[SRC]]{{.*}} : memref<?x?xf32>, vector<8x16xf32>
// CHECK:         %[[M:.*]] = vector.create_mask %[[N]], %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[M]], %[[RD]], %[[Z]]
// CHECK:         return %[[SEL]]
func.func @copy_live(%src: memref<?x?xf32>, %n: index, %pad: f32) -> vector<8x16xf32> {
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

// Copy OUT of the slot -> a transfer_write of the slot value into the target.
// CHECK-LABEL: func.func @copy_out(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[DST:.*]]: memref<8x16xf32>
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         vector.transfer_write %[[V]], %[[DST]]
func.func @copy_out(%v: vector<8x16xf32>, %dst: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %a, %dst : memref<8x16xf32> to memref<8x16xf32>
  return
}

// -----

// Copy between two promotable slots: both promote, the value threads across.
// CHECK-LABEL: func.func @copy_between_slots(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.copy
// CHECK:         return %[[V]]
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

// NEGATIVE: a self-copy references the slot on both sides and is not modeled,
// so the buffer is not promoted (mem2reg leaves the alloca and its transfers;
// canonicalize then folds the self-copy away, but promotion has already bailed).
// CHECK-LABEL: func.func @negative_self_copy(
// CHECK:         memref.alloca
// CHECK:         vector.transfer_read
func.func @negative_self_copy(%v: vector<8x16xf32>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %a, %a : memref<8x16xf32> to memref<8x16xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Copy from a dynamic subview into a dynamic subview of the slot: the source
// subview is read and composed into the slot with the subview mask.
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
