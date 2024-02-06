// RUN: mlir-opt %s -allocate-arm-sme-tiles -convert-arm-sme-to-llvm -cse -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// Test conversion of ArmSME ops to LLVM intrinsics.

//===----------------------------------------------------------------------===//
// arm_sme.load_tile_slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   func.func @arm_sme_load_tile_slice_hor_i8(
// CHECK-SAME:                                              %[[SRC:.*]]: memref<?x?xi8>,
// CHECK-SAME:                                              %[[MASK:.*]]: vector<[16]xi1>,
// CHECK-SAME:                                              %[[TILE_SLICE_INDEX:.*]]: index)
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[SRC]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[C0_I64:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to i64
// CHECK:           %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[STRIDE:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[OFFSET:.*]] = llvm.mul %[[C0_I64]], %[[STRIDE]]  : i64
// CHECK:           %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[TILE_SLICE_INDEX_I32:.*]] = arith.index_castui %[[TILE_SLICE_INDEX]] : index to i32
// CHECK:           "arm_sme.intr.ld1b.horiz"(%[[MASK]], %[[GEP]], %[[TILE_SLICE_INDEX_I32]]) <{tile_id = 0 : i32}> : (vector<[16]xi1>, !llvm.ptr, i32) -> ()
// CHECK:           return
// CHECK:         }
func.func @arm_sme_load_tile_slice_hor_i8(%src : memref<?x?xi8>, %mask : vector<[16]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_hor_i16
// CHECK: "arm_sme.intr.ld1h.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_hor_i16(%src : memref<?x?xi16>, %mask : vector<[8]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_hor_i32
// CHECK: "arm_sme.intr.ld1w.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_hor_i32(%src : memref<?x?xi32>, %mask : vector<[4]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_hor_i64
// CHECK: "arm_sme.intr.ld1d.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_hor_i64(%src : memref<?x?xi64>, %mask : vector<[2]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_hor_i128
// CHECK: "arm_sme.intr.ld1q.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_hor_i128(%src : memref<?x?xi128>, %mask : vector<[1]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_hor_f16
// CHECK: "arm_sme.intr.ld1h.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_hor_f16(%src : memref<?x?xf16>, %mask : vector<[8]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_hor_bf16
// CHECK: "arm_sme.intr.ld1h.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_hor_bf16(%src : memref<?x?xbf16>, %mask : vector<[8]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_hor_f32
// CHECK: "arm_sme.intr.ld1w.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_hor_f32(%src : memref<?x?xf32>, %mask : vector<[4]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_hor_f64
// CHECK: "arm_sme.intr.ld1d.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_hor_f64(%src : memref<?x?xf64>, %mask : vector<[2]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_i8
// CHECK: "arm_sme.intr.ld1b.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[16]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_i8(%src : memref<?x?xi8>, %mask : vector<[16]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_i16
// CHECK: "arm_sme.intr.ld1h.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_i16(%src : memref<?x?xi16>, %mask : vector<[8]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_i32
// CHECK: "arm_sme.intr.ld1w.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_i32(%src : memref<?x?xi32>, %mask : vector<[4]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_i64
// CHECK: "arm_sme.intr.ld1d.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_i64(%src : memref<?x?xi64>, %mask : vector<[2]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_i128
// CHECK: "arm_sme.intr.ld1q.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_i128(%src : memref<?x?xi128>, %mask : vector<[1]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_f16
// CHECK: "arm_sme.intr.ld1h.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_f16(%src : memref<?x?xf16>, %mask : vector<[8]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_bf16
// CHECK: "arm_sme.intr.ld1h.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_bf16(%src : memref<?x?xbf16>, %mask : vector<[8]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_f32
// CHECK: "arm_sme.intr.ld1w.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_f32(%src : memref<?x?xf32>, %mask : vector<[4]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_load_tile_slice_ver_f64
// CHECK: "arm_sme.intr.ld1d.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_load_tile_slice_ver_f64(%src : memref<?x?xf64>, %mask : vector<[2]xi1>, %tile_slice_index : index) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.store_tile_slice
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:   func.func @arm_sme_store_tile_slice_hor_i8(
// CHECK-SAME:                                               %[[TILE_SLICE_INDEX:.*]]: index,
// CHECK-SAME:                                               %[[MASK:.*]]: vector<[16]xi1>,
// CHECK-SAME:                                               %[[DEST:.*]]: memref<?x?xi8>)
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[DEST]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[C0_I64:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to i64
// CHECK:           %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[STRIDE:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[OFFSET:.*]] = llvm.mul %[[C0_I64]], %[[STRIDE]]  : i64
// CHECK:           %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[TILE_SLICE_INDEX_I32:.*]] = arith.index_castui %[[TILE_SLICE_INDEX]] : index to i32
// CHECK:           "arm_sme.intr.st1b.horiz"(%[[MASK]], %[[GEP]], %[[TILE_SLICE_INDEX_I32]]) <{tile_id = 0 : i32}> : (vector<[16]xi1>, !llvm.ptr, i32) -> ()
// CHECK:           return
// CHECK:         }
func.func @arm_sme_store_tile_slice_hor_i8(%tile_slice_index : index,  %mask : vector<[16]xi1>, %dest : memref<?x?xi8>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_hor_i16
// CHECK: "arm_sme.intr.st1h.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_hor_i16(%tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xi16>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_hor_i32
// CHECK: "arm_sme.intr.st1w.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_hor_i32(%tile_slice_index : index, %mask : vector<[4]xi1>, %dest : memref<?x?xi32>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_hor_i64
// CHECK: "arm_sme.intr.st1d.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_hor_i64(%tile_slice_index : index, %mask : vector<[2]xi1>, %dest : memref<?x?xi64>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_hor_i128
// CHECK: "arm_sme.intr.st1q.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_hor_i128(%tile_slice_index : index, %mask : vector<[1]xi1>, %dest : memref<?x?xi128>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_hor_f16
// CHECK: "arm_sme.intr.st1h.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_hor_f16(%tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xf16>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_hor_bf16
// CHECK: "arm_sme.intr.st1h.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_hor_bf16(%tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xbf16>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_hor_f32
// CHECK: "arm_sme.intr.st1w.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_hor_f32(%tile_slice_index : index, %mask : vector<[4]xi1>, %dest : memref<?x?xf32>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_hor_f64
// CHECK: "arm_sme.intr.st1d.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_hor_f64(%tile_slice_index : index, %mask : vector<[2]xi1>, %dest : memref<?x?xf64>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_i8
// CHECK: "arm_sme.intr.st1b.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[16]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_i8(%tile_slice_index : index, %mask : vector<[16]xi1>, %dest : memref<?x?xi8>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_i16
// CHECK: "arm_sme.intr.st1h.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_i16(%tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xi16>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_i32
// CHECK: "arm_sme.intr.st1w.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_i32(%tile_slice_index : index, %mask : vector<[4]xi1>, %dest : memref<?x?xi32>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_i64
// CHECK: "arm_sme.intr.st1d.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_i64(%tile_slice_index : index, %mask : vector<[2]xi1>, %dest : memref<?x?xi64>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_i128
// CHECK: "arm_sme.intr.st1q.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_i128(%tile_slice_index : index, %mask : vector<[1]xi1>, %dest : memref<?x?xi128>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_f16
// CHECK: "arm_sme.intr.st1h.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_f16(%tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xf16>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_bf16
// CHECK: "arm_sme.intr.st1h.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_bf16(%tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xbf16>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_f32
// CHECK: "arm_sme.intr.st1w.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_f32(%tile_slice_index : index, %mask : vector<[4]xi1>, %dest : memref<?x?xf32>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_store_tile_slice_ver_f64
// CHECK: "arm_sme.intr.st1d.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, !llvm.ptr, i32) -> ()
func.func @arm_sme_store_tile_slice_ver_f64(%tile_slice_index : index, %mask : vector<[2]xi1>, %dest : memref<?x?xf64>) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.move_vector_to_tile_slice
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @arm_sme_move_vector_to_tile_slice_hor_i32
// CHECK: "arm_sme.intr.write.horiz"({{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[4]xi1>, vector<[4]xi32>) -> ()
func.func @arm_sme_move_vector_to_tile_slice_hor_i32(%vector : vector<[4]xi32>, %tile_slice_index : index) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[4]xi32> into vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_move_vector_to_tile_slice_ver_bf16
// CHECK: "arm_sme.intr.write.vert"({{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[8]xi1>, vector<[8]xbf16>) -> ()
func.func @arm_sme_move_vector_to_tile_slice_ver_bf16(%vector : vector<[8]xbf16>, %tile_slice_index : index) -> () {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index layout<vertical> : vector<[8]xbf16> into vector<[8]x[8]xbf16>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.move_tile_slice_to_vector
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_i8
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[16]xi8>, vector<[16]xi1>, i32) -> vector<[16]xi8>
func.func @arm_sme_move_tile_slice_to_vector_i8(%tile_slice_index : index) -> vector<[16]xi8> {
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[16]xi8> from vector<[16]x[16]xi8>
  return %slice : vector<[16]xi8>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_i16
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi16>, vector<[8]xi1>, i32) -> vector<[8]xi16>
func.func @arm_sme_move_tile_slice_to_vector_i16(%tile_slice_index : index) -> vector<[8]xi16> {
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[8]xi16> from vector<[8]x[8]xi16>
  return %slice : vector<[8]xi16>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_i32
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>
func.func @arm_sme_move_tile_slice_to_vector_i32(%tile_slice_index : index) -> vector<[4]xi32> {
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[4]xi32> from vector<[4]x[4]xi32>
  return %slice : vector<[4]xi32>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_i64
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi64>, vector<[2]xi1>, i32) -> vector<[2]xi64>
func.func @arm_sme_move_tile_slice_to_vector_i64(%tile_slice_index : index) -> vector<[2]xi64> {
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[2]xi64> from vector<[2]x[2]xi64>
  return %slice : vector<[2]xi64>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_i128
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi128>, vector<[1]xi1>, i32) -> vector<[1]xi128>
func.func @arm_sme_move_tile_slice_to_vector_i128(%tile_slice_index : index) -> vector<[1]xi128> {
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[1]xi128> from vector<[1]x[1]xi128>
  return %slice : vector<[1]xi128>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_f16
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xf16>, vector<[8]xi1>, i32) -> vector<[8]xf16>
func.func @arm_sme_move_tile_slice_to_vector_f16(%tile_slice_index : index) -> vector<[8]xf16> {
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[8]xf16> from vector<[8]x[8]xf16>
  return %slice : vector<[8]xf16>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_bf16
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xbf16>, vector<[8]xi1>, i32) -> vector<[8]xbf16>
func.func @arm_sme_move_tile_slice_to_vector_bf16(%tile_slice_index : index) -> vector<[8]xbf16> {
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[8]xbf16> from vector<[8]x[8]xbf16>
  return %slice : vector<[8]xbf16>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_f32
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
func.func @arm_sme_move_tile_slice_to_vector_f32(%tile_slice_index : index) -> vector<[4]xf32> {
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[4]xf32> from vector<[4]x[4]xf32>
  return %slice : vector<[4]xf32>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_f64
// CHECK: "arm_sme.intr.read.horiz"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xf64>, vector<[2]xi1>, i32) -> vector<[2]xf64>
func.func @arm_sme_move_tile_slice_to_vector_f64(%tile_slice_index : index) -> vector<[2]xf64> {
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[2]xf64> from vector<[2]x[2]xf64>
  return %slice : vector<[2]xf64>
}

// -----

// CHECK-LABEL: @arm_sme_move_tile_slice_to_vector_ver_i128
// CHECK: "arm_sme.intr.read.vert"({{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi128>, vector<[1]xi1>, i32) -> vector<[1]xi128>
func.func @arm_sme_move_tile_slice_to_vector_ver_i128(%tile_slice_index : index) -> vector<[1]xi128> {
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] layout<vertical> : vector<[1]xi128> from vector<[1]x[1]xi128>
  return %slice : vector<[1]xi128>
}

//===----------------------------------------------------------------------===//
// arm_sme.streaming_vl
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @arm_sme_streaming_vl_bytes
// CHECK: %[[COUNT:.*]] = "arm_sme.intr.cntsb"() : () -> i64
// CHECK: %[[INDEX_COUNT:.*]] = arith.index_cast %[[COUNT]] : i64 to index
// CHECK: return %[[INDEX_COUNT]] : index
func.func @arm_sme_streaming_vl_bytes() -> index {
  %svl_b = arm_sme.streaming_vl <byte>
  return %svl_b : index
}

// -----

// CHECK-LABEL: @arm_sme_streaming_vl_half_words
// CHECK: "arm_sme.intr.cntsh"() : () -> i64
func.func @arm_sme_streaming_vl_half_words() -> index {
  %svl_h = arm_sme.streaming_vl <half>
  return %svl_h : index
}

// -----

// CHECK-LABEL: @arm_sme_streaming_vl_words
// CHECK: "arm_sme.intr.cntsw"() : () -> i64
func.func @arm_sme_streaming_vl_words() -> index {
  %svl_w = arm_sme.streaming_vl <word>
  return %svl_w : index
}

// -----

// CHECK-LABEL: @arm_sme_streaming_vl_double_words
// CHECK: "arm_sme.intr.cntsd"() : () -> i64
func.func @arm_sme_streaming_vl_double_words() -> index {
  %svl_d = arm_sme.streaming_vl <double>
  return %svl_d : index
}

//===----------------------------------------------------------------------===//
// arm_sme.fmopa_2way
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: arm_sme_fmopa_2way_f16f16_to_f32
// CHECK: "arm_sme.intr.mopa.wide"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>) -> ()
func.func @arm_sme_fmopa_2way_f16f16_to_f32(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>) -> vector<[4]x[4]xf32> {
  %result = arm_sme.fmopa_2way %vecA, %vecB : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: arm_sme_fmopa_2way_bf16bf16_to_f32
// CHECK: "arm_sme.intr.mopa.wide"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>) -> ()
func.func @arm_sme_fmopa_2way_bf16bf16_to_f32(%vecA: vector<[8]xbf16>, %vecB: vector<[8]xbf16>) -> vector<[4]x[4]xf32> {
  %result = arm_sme.fmopa_2way %vecA, %vecB : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

//===----------------------------------------------------------------------===//
// arm_sme.fmops_2way
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: arm_sme_fmops_2way_f16f16_to_f32
// CHECK: "arm_sme.intr.mops.wide"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>) -> ()
func.func @arm_sme_fmops_2way_f16f16_to_f32(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>) -> vector<[4]x[4]xf32> {
  %result = arm_sme.fmops_2way %vecA, %vecB : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: arm_sme_fmops_2way_bf16bf16_to_f32
// CHECK: "arm_sme.intr.mops.wide"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>) -> ()
func.func @arm_sme_fmops_2way_bf16bf16_to_f32(%vecA: vector<[8]xbf16>, %vecB: vector<[8]xbf16>) -> vector<[4]x[4]xf32> {
  %result = arm_sme.fmops_2way %vecA, %vecB : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

//===----------------------------------------------------------------------===//
// arm_sme.smopa_2way
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: arm_sme_smopa_2way_i16i16_to_i32
// CHECK: "arm_sme.intr.smopa.za32"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
func.func @arm_sme_smopa_2way_i16i16_to_i32(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[4]x[4]xi32> {
  %result = arm_sme.smopa_2way %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  return %result : vector<[4]x[4]xi32>
}

//===----------------------------------------------------------------------===//
// arm_sme.smops_2way
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: arm_sme_smops_2way_i16i16_to_i32
// CHECK: "arm_sme.intr.smops.za32"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
func.func @arm_sme_smops_2way_i16i16_to_i32(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[4]x[4]xi32> {
  %result = arm_sme.smops_2way %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  return %result : vector<[4]x[4]xi32>
}

//===----------------------------------------------------------------------===//
// arm_sme.umopa_2way
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: arm_sme_umopa_2way_i16i16_to_i32
// CHECK: "arm_sme.intr.umopa.za32"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
func.func @arm_sme_umopa_2way_i16i16_to_i32(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[4]x[4]xi32> {
  %result = arm_sme.umopa_2way %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  return %result : vector<[4]x[4]xi32>
}

//===----------------------------------------------------------------------===//
// arm_sme.umops_2way
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: arm_sme_umops_2way_i16i16_to_i32
// CHECK: "arm_sme.intr.umops.za32"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
func.func @arm_sme_umops_2way_i16i16_to_i32(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[4]x[4]xi32> {
  %result = arm_sme.umops_2way %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  return %result : vector<[4]x[4]xi32>
}
