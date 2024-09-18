// RUN: mlir-opt %s -convert-vector-to-arm-sme -split-input-file -allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// vector.transfer_read
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @transfer_read_2d_i8
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi8>, vector<[16]x[16]xi8>
func.func @transfer_read_2d_i8(%src : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i8
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xi8>, vector<[16]x[16]xi8>
  "prevent.dce"(%0) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_i16
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi16>, vector<[8]x[8]xi16>
func.func @transfer_read_2d_i16(%src : memref<?x?xi16>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i16
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xi16>, vector<[8]x[8]xi16>
  "prevent.dce"(%0) : (vector<[8]x[8]xi16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_i32
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi32>, vector<[4]x[4]xi32>
func.func @transfer_read_2d_i32(%src : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i32
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xi32>, vector<[4]x[4]xi32>
  "prevent.dce"(%0) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_i64
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi64>, vector<[2]x[2]xi64>
func.func @transfer_read_2d_i64(%src : memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i64
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xi64>, vector<[2]x[2]xi64>
  "prevent.dce"(%0) : (vector<[2]x[2]xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_i128
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi128>, vector<[1]x[1]xi128>
func.func @transfer_read_2d_i128(%src : memref<?x?xi128>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i128
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xi128>, vector<[1]x[1]xi128>
  "prevent.dce"(%0) : (vector<[1]x[1]xi128>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_f16
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xf16>, vector<[8]x[8]xf16>
func.func @transfer_read_2d_f16(%src : memref<?x?xf16>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xf16>, vector<[8]x[8]xf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_bf16
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
func.func @transfer_read_2d_bf16(%src : memref<?x?xbf16>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : bf16
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xbf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_f32
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @transfer_read_2d_f32(%src : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>
  "prevent.dce"(%0) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_f64
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xf64>, vector<[2]x[2]xf64>
func.func @transfer_read_2d_f64(%src : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f64
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xf64>, vector<[2]x[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_with_mask_i16
// CHECK: arm_sme.tile_load %{{.*}}[{{.*}}], {{.*}}, {{.*}} : memref<?x?xi16>, vector<[8]x[8]xi16>
func.func @transfer_read_2d_with_mask_i16(%src : memref<?x?xi16>, %mask : vector<[8]x[8]xi1>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i16
  %0 = vector.transfer_read %src[%c0, %c0], %pad, %mask {in_bounds = [true, true]} : memref<?x?xi16>, vector<[8]x[8]xi16>
  "prevent.dce"(%0) : (vector<[8]x[8]xi16>) -> ()
  return
}

// -----

/// in-flight transpose

// CHECK-LABEL: @transfer_read_2d_transpose_i8
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi8>, vector<[16]x[16]xi8>
func.func @transfer_read_2d_transpose_i8(%src : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i8
  %0 = vector.transfer_read %src[%c0, %c0], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<?x?xi8>, vector<[16]x[16]xi8>
  "prevent.dce"(%0) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d_transpose_with_mask_f32
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @transfer_read_2d_transpose_with_mask_f32(%src : memref<?x?xf32>, %mask : vector<[4]x[4]xi1>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %src[%c0, %c0], %pad, %mask {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>
  "prevent.dce"(%0) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: @fold_transpose_into_load
// CHECK-NOT: arm_sme.tile_store
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK-NOT: arm_sme.tile_store
func.func @fold_transpose_into_load(%src : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>
  %1 = vector.transpose %0, [1, 0] : vector<[4]x[4]xf32> to vector<[4]x[4]xf32>
  "prevent.dce"(%1) : (vector<[4]x[4]xf32>) -> ()
}

// -----

/// Transposes with more than a single use cannot be folded into load and will
/// instead be transposed via memory.

// CHECK-LABEL: @fold_transpose_into_load_multi_use
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK: %[[TILE_TRANSPOSED_VIA_MEM:.*]] = arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK: "prevent.dce"(%[[TILE_TRANSPOSED_VIA_MEM]]) : (vector<[4]x[4]xf32>) -> ()
func.func @fold_transpose_into_load_multi_use(%src : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>
  "test.some_use"(%0) : (vector<[4]x[4]xf32>) -> ()
  %1 = vector.transpose %0, [1, 0] : vector<[4]x[4]xf32> to vector<[4]x[4]xf32>
  "prevent.dce"(%1) : (vector<[4]x[4]xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// vector.transfer_write
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @transfer_write_2d_i8(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[16]x[16]xi8>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xi8>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi8>, vector<[16]x[16]xi8>
func.func @transfer_write_2d_i8(%vector : vector<[16]x[16]xi8>, %dest : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_i16(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[8]x[8]xi16>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xi16>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi16>, vector<[8]x[8]xi16>
func.func @transfer_write_2d_i16(%vector : vector<[8]x[8]xi16>, %dest : memref<?x?xi16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xi16>, memref<?x?xi16>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_i32(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[4]x[4]xi32>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xi32>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi32>, vector<[4]x[4]xi32>
func.func @transfer_write_2d_i32(%vector : vector<[4]x[4]xi32>, %dest : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[4]x[4]xi32>, memref<?x?xi32>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_i64(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[2]x[2]xi64>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xi64>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi64>, vector<[2]x[2]xi64>
func.func @transfer_write_2d_i64(%vector : vector<[2]x[2]xi64>, %dest : memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[2]x[2]xi64>, memref<?x?xi64>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_f16(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[8]x[8]xf16>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xf16>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xf16>, vector<[8]x[8]xf16>
func.func @transfer_write_2d_f16(%vector : vector<[8]x[8]xf16>, %dest : memref<?x?xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xf16>, memref<?x?xf16>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_bf16(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[8]x[8]xbf16>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xbf16>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
func.func @transfer_write_2d_bf16(%vector : vector<[8]x[8]xbf16>, %dest : memref<?x?xbf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xbf16>, memref<?x?xbf16>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_f32(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xf32>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @transfer_write_2d_f32(%vector : vector<[4]x[4]xf32>, %dest : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_f64(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[2]x[2]xf64>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xf64>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xf64>, vector<[2]x[2]xf64>
func.func @transfer_write_2d_f64(%vector : vector<[2]x[2]xf64>, %dest : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[2]x[2]xf64>, memref<?x?xf64>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_with_mask_f64(
// CHECK-SAME:                                             %[[VECTOR:.*]]: vector<[2]x[2]xf64>,
// CHECK-SAME:                                             %[[DEST:.*]]: memref<?x?xf64>,
// CHECK-SAME:                                             %[[MASK:.*]]: vector<[2]x[2]xi1>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]], %[[MASK]] : memref<?x?xf64>, vector<[2]x[2]xf64>
func.func @transfer_write_2d_with_mask_f64(%vector : vector<[2]x[2]xf64>, %dest : memref<?x?xf64>, %mask : vector<[2]x[2]xi1>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[2]x[2]xf64>, memref<?x?xf64>
  return
}

// -----

/// in-flight transpose via vertical store.

// CHECK-LABEL: func.func @transfer_write_2d_transpose_i64(
// CHECK-SAME:                                             %[[VECTOR:.*]]: vector<[2]x[2]xi64>,
// CHECK-SAME:                                             %[[DEST:.*]]: memref<?x?xi64>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] layout<vertical> : memref<?x?xi64>, vector<[2]x[2]xi64>
func.func @transfer_write_2d_transpose_i64(%vector : vector<[2]x[2]xi64>, %dest : memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : vector<[2]x[2]xi64>, memref<?x?xi64>
  return
}

// -----

/// in-flight transpose via vertical store with mask.

// CHECK-LABEL: func.func @transfer_write_2d_transpose_with_mask_bf16(
// CHECK-SAME:                                                        %[[VECTOR:.*]]: vector<[8]x[8]xbf16>,
// CHECK-SAME:                                                        %[[DEST:.*]]: memref<?x?xbf16>,
// CHECK-SAME:                                                        %[[MASK:.*]]: vector<[8]x[8]xi1>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]], %[[MASK]] layout<vertical> : memref<?x?xbf16>, vector<[8]x[8]xbf16>
func.func @transfer_write_2d_transpose_with_mask_bf16(%vector : vector<[8]x[8]xbf16>, %dest : memref<?x?xbf16>, %mask : vector<[8]x[8]xi1>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0], %mask {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : vector<[8]x[8]xbf16>, memref<?x?xbf16>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_slice(
// CHECK-SAME:                                  %[[VECTOR:.*]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                  %[[DEST:.*]]: memref<?x?xf32>,
// CHECK-SAME:                                  %[[INDEX:.*]]: index) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[MASK:.*]] = arith.constant dense<true> : vector<[4]xi1>
// CHECK:         arm_sme.store_tile_slice %[[VECTOR]], %[[INDEX]], %[[MASK]], %[[DEST]][%[[INDEX]], %[[C0]]] : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
func.func @transfer_write_slice(%vector: vector<[4]x[4]xf32>, %dest : memref<?x?xf32>, %slice_index: index) {
  %c0 = arith.constant 0 : index
  %slice = vector.extract %vector[%slice_index] : vector<[4]xf32> from vector<[4]x[4]xf32>
  vector.transfer_write %slice, %dest[%slice_index, %c0] { in_bounds = [true] }: vector<[4]xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_slice_with_mask(
// CHECK-SAME:                                            %[[VECTOR:.*]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                            %[[DEST:.*]]: memref<?x?xf32>,
// CHECK-SAME:                                            %[[MASK:.*]]: vector<[4]xi1>,
// CHECK-SAME:                                            %[[INDEX:.*]]: index) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.store_tile_slice %[[VECTOR]], %[[INDEX]], %[[MASK]], %[[DEST]][%[[INDEX]], %[[C0]]] : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
func.func @transfer_write_slice_with_mask(%vector: vector<[4]x[4]xf32>, %dest : memref<?x?xf32>, %mask: vector<[4]xi1>, %slice_index: index) {
  %c0 = arith.constant 0 : index
  %slice = vector.extract %vector[%slice_index] : vector<[4]xf32> from vector<[4]x[4]xf32>
  vector.transfer_write %slice, %dest[%slice_index, %c0], %mask { in_bounds = [true] }: vector<[4]xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_vertical_slice
// CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical>
func.func @transfer_write_vertical_slice(%vector: vector<[4]x[4]xf32>, %dest : memref<?x?xf32>, %slice_index: index) {
  %c0 = arith.constant 0 : index
   %slice = arm_sme.extract_tile_slice %vector[%slice_index] layout<vertical>
            : vector<[4]xf32> from vector<[4]x[4]xf32>
  vector.transfer_write %slice, %dest[%slice_index, %c0] { in_bounds = [true] }: vector<[4]xf32>, memref<?x?xf32>
  return
}

//===----------------------------------------------------------------------===//
// vector.broadcast
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:   func.func @broadcast_vec2d_from_i32(
// CHECK-SAME:                                        %[[SRC:.*]]: i32) {
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[SRC_1D:.*]] = vector.broadcast %[[SRC]] : i32 to vector<[4]xi32>
// CHECK: %[[INIT_TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
// CHECK: %[[VSCALE:.*]] = vector.vscale
// CHECK: %[[NUM_TILE_SLICES:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
// CHECK: %[[TILE:.*]] = scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[NUM_TILE_SLICES]] step %[[C1]] iter_args(%[[CURRENT_TILE:.*]] = %[[INIT_TILE]]) -> (vector<[4]x[4]xi32>) {
// CHECK:   %[[NEW_TILE:.*]] = arm_sme.insert_tile_slice %[[SRC_1D]], %[[CURRENT_TILE]][%[[TILE_SLICE_INDEX]]] : vector<[4]xi32> into vector<[4]x[4]xi32>
// CHECK:   scf.yield %[[NEW_TILE]] : vector<[4]x[4]xi32>
// CHECK: "prevent.dce"(%[[TILE]]) : (vector<[4]x[4]xi32>) -> ()
func.func @broadcast_vec2d_from_i32(%arg0: i32) {
  %0 = vector.broadcast %arg0 : i32 to vector<[4]x[4]xi32>
  "prevent.dce"(%0) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL:   func.func @broadcast_vec2d_from_vec0d(
// CHECK-SAME:                                          %[[SRC:.*]]: vector<f32>) {
// CHECK: %[[SRC_1D:.*]] = vector.broadcast %[[SRC]] : vector<f32> to vector<[4]xf32>
// CHECK: scf.for
// CHECK:   arm_sme.insert_tile_slice %[[SRC_1D]], {{.*}}
func.func @broadcast_vec2d_from_vec0d(%arg0: vector<f32>) {
  %0 = vector.broadcast %arg0 : vector<f32> to vector<[4]x[4]xf32>
  "prevent.dce"(%0) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-LABEL:   func.func @broadcast_vec2d_from_vec1d(
// CHECK-SAME:                                          %[[SRC:.*]]: vector<[8]xi16>) {
// CHECK-NOT: vector.broadcast
// CHECK: scf.for
// CHECK:   arm_sme.insert_tile_slice %[[SRC]], {{.*}}
func.func @broadcast_vec2d_from_vec1d(%arg0: vector<[8]xi16>) {
  %0 = vector.broadcast %arg0 : vector<[8]xi16> to vector<[8]x[8]xi16>
  "prevent.dce"(%0) : (vector<[8]x[8]xi16>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// vector.splat
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:   func.func @splat_vec2d_from_i32(
// CHECK-SAME:      %[[SRC:.*]]: i32) {
// CHECK:   %[[BCST:.*]] = vector.broadcast %[[SRC]] : i32 to vector<[4]xi32>
// CHECK:   arm_sme.get_tile : vector<[4]x[4]xi32>
// CHECK:   %[[VSCALE:.*]] = vector.vscale
// CHECK:   %[[NUM_TILE_SLICES:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
// CHECK:   scf.for {{.*}} to %[[NUM_TILE_SLICES]] {{.*}} {
// CHECK:     arm_sme.insert_tile_slice %[[BCST]], {{.*}} : vector<[4]xi32> into vector<[4]x[4]xi32>
func.func @splat_vec2d_from_i32(%arg0: i32) {
  %0 = vector.splat %arg0 : vector<[4]x[4]xi32>
  "prevent.dce"(%0) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL:   func.func @splat_vec2d_from_f16(
// CHECK-SAME:      %[[SRC:.*]]: f16) {
// CHECK:   %[[BCST:.*]] = vector.broadcast %[[SRC]] : f16 to vector<[8]xf16>
// CHECK:   scf.for
// CHECK:     arm_sme.insert_tile_slice %[[BCST]], {{.*}} : vector<[8]xf16> into vector<[8]x[8]xf16>
func.func @splat_vec2d_from_f16(%arg0: f16) {
  %0 = vector.splat %arg0 : vector<[8]x[8]xf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xf16>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// vector.transpose
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:   func.func @transpose_i8(
// CHECK-SAME:                            %[[TILE:.*]]: vector<[16]x[16]xi8>)
// CHECK-DAG:       %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[VSCALE:.*]] = vector.vscale
// CHECK:           %[[MIN_TILE_SLICES:.*]] = arith.muli %[[VSCALE]], %[[C16]] : index
// CHECK:           %[[NUM_TILE_SLICES:.*]] = memref.alloca(%[[MIN_TILE_SLICES]], %[[MIN_TILE_SLICES]]) : memref<?x?xi8>
// CHECK:           arm_sme.tile_store %[[TILE]], %[[NUM_TILE_SLICES]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi8>, vector<[16]x[16]xi8>
// CHECK:           arm_sme.tile_load %[[NUM_TILE_SLICES]]{{\[}}%[[C0]], %[[C0]]] layout<vertical> : memref<?x?xi8>, vector<[16]x[16]xi8>
func.func @transpose_i8(%arg0: vector<[16]x[16]xi8>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[16]x[16]xi8> to vector<[16]x[16]xi8>
  "prevent.dce"(%0) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_i16
// CHECK: arith.constant 8
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi16>, vector<[8]x[8]xi16>
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi16>, vector<[8]x[8]xi16>
func.func @transpose_i16(%arg0: vector<[8]x[8]xi16>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[8]x[8]xi16> to vector<[8]x[8]xi16>
  "prevent.dce"(%0) : (vector<[8]x[8]xi16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_i32
// CHECK: arith.constant 4
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi32>, vector<[4]x[4]xi32>
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
func.func @transpose_i32(%arg0: vector<[4]x[4]xi32>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[4]x[4]xi32> to vector<[4]x[4]xi32>
  "prevent.dce"(%0) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_i64
// CHECK: arith.constant 2
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi64>, vector<[2]x[2]xi64>
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi64>, vector<[2]x[2]xi64>
func.func @transpose_i64(%arg0: vector<[2]x[2]xi64>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[2]x[2]xi64> to vector<[2]x[2]xi64>
  "prevent.dce"(%0) : (vector<[2]x[2]xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_i128
// CHECK: %[[VSCALE:.*]] = vector.vscale
// CHECK: %[[NUM_TILE_SLICES:.*]] = memref.alloca(%[[VSCALE]], %[[VSCALE]]) : memref<?x?xi128>
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi128>, vector<[1]x[1]xi128>
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi128>, vector<[1]x[1]xi128>
func.func @transpose_i128(%arg0: vector<[1]x[1]xi128>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[1]x[1]xi128> to vector<[1]x[1]xi128>
  "prevent.dce"(%0) : (vector<[1]x[1]xi128>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_f16
// CHECK: arith.constant 8
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf16>, vector<[8]x[8]xf16>
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf16>, vector<[8]x[8]xf16>
func.func @transpose_f16(%arg0: vector<[8]x[8]xf16>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[8]x[8]xf16> to vector<[8]x[8]xf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_bf16
// CHECK: arith.constant 8
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xbf16>, vector<[8]x[8]xbf16>
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xbf16>, vector<[8]x[8]xbf16>
func.func @transpose_bf16(%arg0: vector<[8]x[8]xbf16>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[8]x[8]xbf16> to vector<[8]x[8]xbf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xbf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_f32
// CHECK: arith.constant 4
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @transpose_f32(%arg0: vector<[4]x[4]xf32>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[4]x[4]xf32> to vector<[4]x[4]xf32>
  "prevent.dce"(%0) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_f64
// CHECK: arith.constant 2
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf64>, vector<[2]x[2]xf64>
// CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf64>, vector<[2]x[2]xf64>
func.func @transpose_f64(%arg0: vector<[2]x[2]xf64>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[2]x[2]xf64> to vector<[2]x[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// vector.outerproduct
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_outerproduct_masked_f16
// CHECK-SAME: (%[[LHS:.*]]: vector<[8]xf16>, %[[RHS:.*]]: vector<[8]xf16>, %[[ACC:.*]]: vector<[8]x[8]xf16>, %[[DIM0:.*]]: index, %[[DIM1:.*]]: index
func.func @vector_outerproduct_masked_f16(%lhs : vector<[8]xf16>, %rhs : vector<[8]xf16>, %acc : vector<[8]x[8]xf16>, %dim0 : index, %dim1 : index) {
  %mask = vector.create_mask %dim0, %dim1 : vector<[8]x[8]xi1>
  // CHECK: %[[LHS_MASK:.*]] = vector.create_mask %[[DIM0]] : vector<[8]xi1>
  // CHECK: %[[RHS_MASK:.*]] = vector.create_mask %[[DIM1]] : vector<[8]xi1>
  // CHECK: arm_sme.outerproduct %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[8]xf16>, vector<[8]xf16>
  %result = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[8]xf16>, vector<[8]xf16> } : vector<[8]x[8]xi1> -> vector<[8]x[8]xf16>
  "prevent.dce"(%result) : (vector<[8]x[8]xf16>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_masked_bf16
// CHECK-SAME: (%[[LHS:.*]]: vector<[8]xbf16>, %[[RHS:.*]]: vector<[8]xbf16>, %[[ACC:.*]]: vector<[8]x[8]xbf16>, %[[DIM0:.*]]: index, %[[DIM1:.*]]: index
func.func @vector_outerproduct_masked_bf16(%lhs : vector<[8]xbf16>, %rhs : vector<[8]xbf16>, %acc : vector<[8]x[8]xbf16>, %dim0 : index, %dim1 : index) {
  %mask = vector.create_mask %dim0, %dim1 : vector<[8]x[8]xi1>
  // CHECK: %[[LHS_MASK:.*]] = vector.create_mask %[[DIM0]] : vector<[8]xi1>
  // CHECK: %[[RHS_MASK:.*]] = vector.create_mask %[[DIM1]] : vector<[8]xi1>
  // CHECK: arm_sme.outerproduct %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[8]xbf16>, vector<[8]xbf16>
  %result = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[8]xbf16>, vector<[8]xbf16> } : vector<[8]x[8]xi1> -> vector<[8]x[8]xbf16>
  "prevent.dce"(%result) : (vector<[8]x[8]xbf16>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_masked_f32
// CHECK-SAME: (%[[LHS:.*]]: vector<[4]xf32>, %[[RHS:.*]]: vector<[4]xf32>, %[[ACC:.*]]: vector<[4]x[4]xf32>, %[[DIM0:.*]]: index, %[[DIM1:.*]]: index
func.func @vector_outerproduct_masked_f32(%lhs : vector<[4]xf32>, %rhs : vector<[4]xf32>, %acc : vector<[4]x[4]xf32>, %dim0 : index, %dim1 : index) {
  %mask = vector.create_mask %dim0, %dim1 : vector<[4]x[4]xi1>
  // CHECK: %[[LHS_MASK:.*]] = vector.create_mask %[[DIM0]] : vector<[4]xi1>
  // CHECK: %[[RHS_MASK:.*]] = vector.create_mask %[[DIM1]] : vector<[4]xi1>
  // CHECK: arm_sme.outerproduct %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[4]xf32>, vector<[4]xf32>
  %result = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32> } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>
  "prevent.dce"(%result) : (vector<[4]x[4]xf32>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_masked_f64
// CHECK-SAME: (%[[LHS:.*]]: vector<[2]xf64>, %[[RHS:.*]]: vector<[2]xf64>, %[[ACC:.*]]: vector<[2]x[2]xf64>, %[[DIM0:.*]]: index, %[[DIM1:.*]]: index
func.func @vector_outerproduct_masked_f64(%lhs : vector<[2]xf64>, %rhs : vector<[2]xf64>, %acc : vector<[2]x[2]xf64>, %dim0 : index, %dim1 : index) {
  %mask = vector.create_mask %dim0, %dim1 : vector<[2]x[2]xi1>
  // CHECK: %[[LHS_MASK:.*]] = vector.create_mask %[[DIM0]] : vector<[2]xi1>
  // CHECK: %[[RHS_MASK:.*]] = vector.create_mask %[[DIM1]] : vector<[2]xi1>
  // CHECK: arm_sme.outerproduct %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[2]xf64>, vector<[2]xf64>
  %result = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[2]xf64>, vector<[2]xf64> } : vector<[2]x[2]xi1> -> vector<[2]x[2]xf64>
  "prevent.dce"(%result) : (vector<[2]x[2]xf64>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_f16
// CHECK-SAME: (%[[LHS:.*]]: vector<[8]xf16>, %[[RHS:.*]]: vector<[8]xf16>, %[[ACC:.*]]: vector<[8]x[8]xf16>
func.func @vector_outerproduct_f16(%lhs : vector<[8]xf16>, %rhs : vector<[8]xf16>, %acc : vector<[8]x[8]xf16>) {
  // CHECK: arm_sme.outerproduct %[[LHS]], %[[RHS]] acc(%[[ACC]]) : vector<[8]xf16>, vector<[8]xf16>
  %result = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[8]xf16>, vector<[8]xf16>
  "prevent.dce"(%result) : (vector<[8]x[8]xf16>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_bf16
// CHECK-SAME: (%[[LHS:.*]]: vector<[8]xbf16>, %[[RHS:.*]]: vector<[8]xbf16>, %[[ACC:.*]]: vector<[8]x[8]xbf16>
func.func @vector_outerproduct_bf16(%lhs : vector<[8]xbf16>, %rhs : vector<[8]xbf16>, %acc : vector<[8]x[8]xbf16>) {
  // CHECK: arm_sme.outerproduct %[[LHS]], %[[RHS]] acc(%[[ACC]]) : vector<[8]xbf16>, vector<[8]xbf16>
  %result = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[8]xbf16>, vector<[8]xbf16>
  "prevent.dce"(%result) : (vector<[8]x[8]xbf16>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_f32
// CHECK-SAME: (%[[LHS:.*]]: vector<[4]xf32>, %[[RHS:.*]]: vector<[4]xf32>, %[[ACC:.*]]: vector<[4]x[4]xf32>
func.func @vector_outerproduct_f32(%lhs : vector<[4]xf32>, %rhs : vector<[4]xf32>, %acc : vector<[4]x[4]xf32>) {
  // CHECK: arm_sme.outerproduct %[[LHS]], %[[RHS]] acc(%[[ACC]]) : vector<[4]xf32>, vector<[4]xf32>
  %result = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32>
  "prevent.dce"(%result) : (vector<[4]x[4]xf32>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_f64
// CHECK-SAME: (%[[LHS:.*]]: vector<[2]xf64>, %[[RHS:.*]]: vector<[2]xf64>, %[[ACC:.*]]: vector<[2]x[2]xf64>
func.func @vector_outerproduct_f64(%lhs : vector<[2]xf64>, %rhs : vector<[2]xf64>, %acc : vector<[2]x[2]xf64>) {
  // CHECK: arm_sme.outerproduct %[[LHS]], %[[RHS]] acc(%[[ACC]]) : vector<[2]xf64>, vector<[2]xf64>
  %result = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[2]xf64>, vector<[2]xf64>
  "prevent.dce"(%result) : (vector<[2]x[2]xf64>) -> ()
}

//===----------------------------------------------------------------------===//
// vector.print
//===----------------------------------------------------------------------===//

// -----

func.func @vector_print_tile(%tile: vector<[4]x[4]xf32>)
{
  vector.print %tile : vector<[4]x[4]xf32>
  return
}
// CHECK-LABEL:   func.func @vector_print_tile(
// CHECK-SAME:                                  %[[TILE:.*]]: vector<[4]x[4]xf32>) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[VSCALE:.*]] = vector.vscale
// CHECK-DAG:     %[[NUM_TILE_SLICES:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
// CHECK-NEXT:      scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[NUM_TILE_SLICES]] step %[[C1]] {
// CHECK-NEXT:        %[[TILE_SLICE:.*]] = arm_sme.extract_tile_slice %[[TILE]][%[[TILE_SLICE_INDEX]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
// CHECK-NEXT:        vector.print %[[TILE_SLICE]] : vector<[4]xf32>

//===----------------------------------------------------------------------===//
// vector.load
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_load_i8_with_offset(
// CHECK-SAME:                              %[[MEMREF:.*]]: memref<?x?xi8>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C123:.*]] = arith.constant 123 : index
// CHECK: arm_sme.tile_load %[[MEMREF]][%[[C123]], %[[C0]]] : memref<?x?xi8>, vector<[16]x[16]xi8>
func.func @vector_load_i8_with_offset(%arg0 : memref<?x?xi8>) -> vector<[16]x[16]xi8> {
  %c0 = arith.constant 0 : index
  %c123 = arith.constant 123 : index
  %tile = vector.load %arg0[%c123, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return %tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @vector_load_i8_from_rank_1_memref(
// CHECK-SAME:                                     %[[MEMREF:.*]]: memref<?xi8>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: arm_sme.tile_load %[[MEMREF]][%[[C0]]] : memref<?xi8>, vector<[16]x[16]xi8>
func.func @vector_load_i8_from_rank_1_memref(%arg0 : memref<?xi8>) -> vector<[16]x[16]xi8> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0] : memref<?xi8>, vector<[16]x[16]xi8>
  return %tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @vector_load_i16(
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi16>, vector<[8]x[8]xi16>
func.func @vector_load_i16(%arg0 : memref<?x?xi16>) -> vector<[8]x[8]xi16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return %tile : vector<[8]x[8]xi16>
}

// -----

// CHECK-LABEL: @vector_load_i32(
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi32>, vector<[4]x[4]xi32>
func.func @vector_load_i32(%arg0 : memref<?x?xi32>) -> vector<[4]x[4]xi32> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return %tile : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @vector_load_i64(
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi64>, vector<[2]x[2]xi64>
func.func @vector_load_i64(%arg0 : memref<?x?xi64>) -> vector<[2]x[2]xi64> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return %tile : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @vector_load_f16(
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xf16>, vector<[8]x[8]xf16>
func.func @vector_load_f16(%arg0 : memref<?x?xf16>) -> vector<[8]x[8]xf16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return %tile : vector<[8]x[8]xf16>
}

// -----

// CHECK-LABEL: @vector_load_bf16(
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xbf16>, vector<[8]x[8]xbf16>
func.func @vector_load_bf16(%arg0 : memref<?x?xbf16>) -> vector<[8]x[8]xbf16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return %tile : vector<[8]x[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_load_f32(
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @vector_load_f32(%arg0 : memref<?x?xf32>) -> vector<[4]x[4]xf32> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return %tile : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @vector_load_f64(
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xf64>, vector<[2]x[2]xf64>
func.func @vector_load_f64(%arg0 : memref<?x?xf64>) -> vector<[2]x[2]xf64> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return %tile : vector<[2]x[2]xf64>
}

// -----

// CHECK-LABEL: @vector_load_i128(
// CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi128>, vector<[1]x[1]xi128>
func.func @vector_load_i128(%arg0 : memref<?x?xi128>) -> vector<[1]x[1]xi128> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
  return %tile : vector<[1]x[1]xi128>
}


//===----------------------------------------------------------------------===//
// vector.store
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_store_i8(
// CHECK-SAME:                   %[[MEMREF:.*]]: memref<?x?xi8>) {
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[16]x[16]xi8>
// CHECK: arm_sme.tile_store %[[TILE]], %[[MEMREF]][%[[C0]], %[[C0]]] : memref<?x?xi8>, vector<[16]x[16]xi8>
func.func @vector_store_i8(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: @vector_store_i16
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi16>, vector<[8]x[8]xi16>
func.func @vector_store_i16(%arg0 : memref<?x?xi16>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return
}

// -----

// CHECK-LABEL: @vector_store_i32
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi32>, vector<[4]x[4]xi32>
func.func @vector_store_i32(%arg0 : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @vector_store_i64
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi64>, vector<[2]x[2]xi64>
func.func @vector_store_i64(%arg0 : memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return
}

// -----

// CHECK-LABEL: @vector_store_f16
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf16>, vector<[8]x[8]xf16>
func.func @vector_store_f16(%arg0 : memref<?x?xf16>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return
}

// -----

// CHECK-LABEL: @vector_store_bf16
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xbf16>, vector<[8]x[8]xbf16>
func.func @vector_store_bf16(%arg0 : memref<?x?xbf16>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return
}
// -----

// CHECK-LABEL: @vector_store_f32
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @vector_store_f32(%arg0 : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

// CHECK-LABEL: @vector_store_f64
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf64>, vector<[2]x[2]xf64>
func.func @vector_store_f64(%arg0 : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

// CHECK-LABEL: @vector_store_i128
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi128>, vector<[1]x[1]xi128>
func.func @vector_store_i128(%arg0 : memref<?x?xi128>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
  return
}

//===----------------------------------------------------------------------===//
// vector.insert
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_insert_slice_i32(
// CHECK-SAME:                       %[[SLICE:.*]]: vector<[4]xi32>,
// CHECK-SAME:                       %[[INDEX:.*]]: index)
func.func @vector_insert_slice_i32(%slice: vector<[4]xi32>, %row: index) -> vector<[4]x[4]xi32>{
  // CHECK-NEXT: %[[TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
  // CHECK-NEXT: arm_sme.insert_tile_slice %[[SLICE]], %[[TILE]][%[[INDEX]]] : vector<[4]xi32> into vector<[4]x[4]xi32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[4]xi32> into vector<[4]x[4]xi32>
  return %new_tile : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @vector_insert_slice_i8
func.func @vector_insert_slice_i8(%slice: vector<[16]xi8>, %row: index) -> vector<[16]x[16]xi8> {
  // CHECK: arm_sme.insert_tile_slice %{{.*}} : vector<[16]xi8> into vector<[16]x[16]xi8>
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[16]xi8> into vector<[16]x[16]xi8>
  return %new_tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @vector_insert_slice_i16
func.func @vector_insert_slice_i16(%slice: vector<[8]xi16>, %row: index) -> vector<[8]x[8]xi16> {
  // CHECK: arm_sme.insert_tile_slice %{{.*}} : vector<[8]xi16> into vector<[8]x[8]xi16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[8]xi16> into vector<[8]x[8]xi16>
  return %new_tile : vector<[8]x[8]xi16>
}

// -----

// CHECK-LABEL: @vector_insert_slice_i64
func.func @vector_insert_slice_i64(%slice: vector<[2]xi64>, %row: index) -> vector<[2]x[2]xi64> {
  // CHECK: arm_sme.insert_tile_slice %{{.*}} : vector<[2]xi64> into vector<[2]x[2]xi64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[2]xi64> into vector<[2]x[2]xi64>
  return %new_tile : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @vector_insert_slice_i128
func.func @vector_insert_slice_i128(%slice: vector<[1]xi128>, %row: index) -> vector<[1]x[1]xi128> {
  // CHECK: arm_sme.insert_tile_slice %{{.*}} : vector<[1]xi128> into vector<[1]x[1]xi128>
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[1]xi128> into vector<[1]x[1]xi128>
  return %new_tile : vector<[1]x[1]xi128>
}

// -----

// CHECK-LABEL: @vector_insert_slice_f16
func.func @vector_insert_slice_f16(%slice: vector<[8]xf16>, %row: index) -> vector<[8]x[8]xf16> {
  // CHECK: arm_sme.insert_tile_slice %{{.*}} : vector<[8]xf16> into vector<[8]x[8]xf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[8]xf16> into vector<[8]x[8]xf16>
  return %new_tile : vector<[8]x[8]xf16>
}

// -----

// CHECK-LABEL: @vector_insert_slice_bf16
func.func @vector_insert_slice_bf16(%slice: vector<[8]xbf16>, %row: index) -> vector<[8]x[8]xbf16> {
  // CHECK: arm_sme.insert_tile_slice %{{.*}} : vector<[8]xbf16> into vector<[8]x[8]xbf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[8]xbf16> into vector<[8]x[8]xbf16>
  return %new_tile : vector<[8]x[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_insert_slice_f32
func.func @vector_insert_slice_f32(%slice: vector<[4]xf32>, %row: index) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.insert_tile_slice %{{.*}} : vector<[4]xf32> into vector<[4]x[4]xf32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[4]xf32> into vector<[4]x[4]xf32>
  return %new_tile : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @vector_insert_slice_f64
func.func @vector_insert_slice_f64(%slice: vector<[2]xf64>, %row: index) -> vector<[2]x[2]xf64> {
  // CHECK: arm_sme.insert_tile_slice %{{.*}} : vector<[2]xf64> into vector<[2]x[2]xf64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[2]xf64> into vector<[2]x[2]xf64>
  return %new_tile : vector<[2]x[2]xf64>
}

// -----

// CHECK-LABEL: @vector_insert_element_i32(
// CHECK-SAME:                         %[[EL:.*]]: i32,
// CHECK-SAME:                         %[[ROW:.*]]: index,
// CHECK-SAME:                         %[[COL:.*]]: index)
func.func @vector_insert_element_i32(%el: i32, %row: index, %col: index) -> vector<[4]x[4]xi32> {
  // CHECK-NEXT: %[[TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = arm_sme.extract_tile_slice %[[TILE]][%[[ROW]]] : vector<[4]xi32> from vector<[4]x[4]xi32>
  // CHECK-NEXT: %[[NEW_SLICE:.*]] = vector.insert %[[EL]], %[[SLICE]] [%[[COL]]] : i32 into vector<[4]xi32>
  // CHECK-NEXT: arm_sme.insert_tile_slice %[[NEW_SLICE]], %[[TILE]][%[[ROW]]] : vector<[4]xi32> into vector<[4]x[4]xi32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %new_tile = vector.insert %el, %tile[%row, %col] : i32 into vector<[4]x[4]xi32>
  return %new_tile : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @vector_insert_element_i8
func.func @vector_insert_element_i8(%el: i8, %row: index, %col: index) -> vector<[16]x[16]xi8> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[16]x[16]xi8>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]]{{.*}} : vector<[16]xi8> from vector<[16]x[16]xi8>
  // CHECK: arm_sme.insert_tile_slice %{{.*}}, %[[TILE]][%{{.*}}] : vector<[16]xi8> into vector<[16]x[16]xi8>
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %new_tile = vector.insert %el, %tile[%row, %col] : i8 into vector<[16]x[16]xi8>
  return %new_tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @vector_insert_element_i16
func.func @vector_insert_element_i16(%el: i16, %row: index, %col: index) -> vector<[8]x[8]xi16> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[8]x[8]xi16>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]]{{.*}} : vector<[8]xi16> from vector<[8]x[8]xi16>
  // CHECK: arm_sme.insert_tile_slice %{{.*}}, %[[TILE]][%{{.*}}] : vector<[8]xi16> into vector<[8]x[8]xi16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %new_tile = vector.insert %el, %tile[%row, %col] : i16 into vector<[8]x[8]xi16>
  return %new_tile : vector<[8]x[8]xi16>
}

// -----

// CHECK-LABEL: @vector_insert_element_i64
func.func @vector_insert_element_i64(%el: i64, %row: index, %col: index) -> vector<[2]x[2]xi64> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]]{{.*}} : vector<[2]xi64> from vector<[2]x[2]xi64>
  // CHECK: arm_sme.insert_tile_slice %{{.*}}, %[[TILE]][%{{.*}}] : vector<[2]xi64> into vector<[2]x[2]xi64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %new_tile = vector.insert %el, %tile[%row, %col] : i64 into vector<[2]x[2]xi64>
  return %new_tile : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @vector_insert_element_i128
func.func @vector_insert_element_i128(%el: i128, %row: index, %col: index) -> vector<[1]x[1]xi128> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]]{{.*}} : vector<[1]xi128> from vector<[1]x[1]xi128>
  // CHECK: arm_sme.insert_tile_slice %{{.*}}, %[[TILE]][%{{.*}}] : vector<[1]xi128> into vector<[1]x[1]xi128>
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %new_tile = vector.insert %el, %tile[%row, %col] : i128 into vector<[1]x[1]xi128>
  return %new_tile : vector<[1]x[1]xi128>
}

// -----

// CHECK-LABEL: @vector_insert_element_f16
func.func @vector_insert_element_f16(%el: f16, %row: index, %col: index) -> vector<[8]x[8]xf16> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[8]x[8]xf16>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]]{{.*}} : vector<[8]xf16> from vector<[8]x[8]xf16>
  // CHECK: arm_sme.insert_tile_slice %{{.*}}, %[[TILE]][%{{.*}}] : vector<[8]xf16> into vector<[8]x[8]xf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %new_tile = vector.insert %el, %tile[%row, %col] : f16 into vector<[8]x[8]xf16>
  return %new_tile : vector<[8]x[8]xf16>
}

// -----

// CHECK-LABEL: @vector_insert_element_bf16
func.func @vector_insert_element_bf16(%el: bf16, %row: index, %col: index) -> vector<[8]x[8]xbf16> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[8]x[8]xbf16>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]]{{.*}} : vector<[8]xbf16> from vector<[8]x[8]xbf16>
  // CHECK: arm_sme.insert_tile_slice %{{.*}}, %[[TILE]][%{{.*}}] : vector<[8]xbf16> into vector<[8]x[8]xbf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %new_tile = vector.insert %el, %tile[%row, %col] : bf16 into vector<[8]x[8]xbf16>
  return %new_tile : vector<[8]x[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_insert_element_f32
func.func @vector_insert_element_f32(%el: f32, %row: index, %col: index) -> vector<[4]x[4]xf32> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xf32>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]]{{.*}} : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK: arm_sme.insert_tile_slice %{{.*}}, %[[TILE]][%{{.*}}] : vector<[4]xf32> into vector<[4]x[4]xf32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %new_tile = vector.insert %el, %tile[%row, %col] : f32 into vector<[4]x[4]xf32>
  return %new_tile : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @vector_insert_element_f64
func.func @vector_insert_element_f64(%el: f64, %row: index, %col: index) -> vector<[2]x[2]xf64> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[2]x[2]xf64>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]]{{.*}} : vector<[2]xf64> from vector<[2]x[2]xf64>
  // CHECK: arm_sme.insert_tile_slice %{{.*}}, %[[TILE]][%{{.*}}] : vector<[2]xf64> into vector<[2]x[2]xf64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %new_tile = vector.insert %el, %tile[%row, %col] : f64 into vector<[2]x[2]xf64>
  return %new_tile : vector<[2]x[2]xf64>
}

//===----------------------------------------------------------------------===//
// vector.extract --> arm_sme.extract_tile_slice
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_extract_slice_i32(
// CHECK-SAME:                            %[[INDEX:.*]]: index)
func.func @vector_extract_slice_i32(%row: index) -> vector<[4]xi32> {
  // CHECK: %[[TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
  // CHECK: arm_sme.extract_tile_slice %[[TILE]][%[[INDEX]]] : vector<[4]xi32> from vector<[4]x[4]xi32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %slice = vector.extract %tile[%row] : vector<[4]xi32> from vector<[4]x[4]xi32>
  return %slice : vector<[4]xi32>
}

// -----

// CHECK-LABEL: @vector_extract_slice_i8
func.func @vector_extract_slice_i8(%row: index) -> vector<[16]xi8> {
  // CHECK: arm_sme.extract_tile_slice {{.*}} : vector<[16]xi8> from vector<[16]x[16]xi8>
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %slice = vector.extract %tile[%row] : vector<[16]xi8> from vector<[16]x[16]xi8>
  return %slice : vector<[16]xi8>
}

// -----

// CHECK-LABEL: @vector_extract_slice_i16
func.func @vector_extract_slice_i16(%row: index) -> vector<[8]xi16> {
  // CHECK: arm_sme.extract_tile_slice {{.*}} : vector<[8]xi16> from vector<[8]x[8]xi16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %slice = vector.extract %tile[%row] : vector<[8]xi16> from vector<[8]x[8]xi16>
  return %slice : vector<[8]xi16>
}

// -----

// CHECK-LABEL: @vector_extract_slice_i64
func.func @vector_extract_slice_i64(%row: index) -> vector<[2]xi64> {
  // CHECK: arm_sme.extract_tile_slice {{.*}} : vector<[2]xi64> from vector<[2]x[2]xi64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %slice = vector.extract %tile[%row] : vector<[2]xi64> from vector<[2]x[2]xi64>
  return %slice : vector<[2]xi64>
}

// -----

// CHECK-LABEL: @vector_extract_slice_i128
func.func @vector_extract_slice_i128(%row: index) -> vector<[1]xi128> {
  // CHECK: arm_sme.extract_tile_slice {{.*}} : vector<[1]xi128> from vector<[1]x[1]xi128>
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %slice = vector.extract %tile[%row] : vector<[1]xi128> from vector<[1]x[1]xi128>
  return %slice : vector<[1]xi128>
}

// -----

// CHECK-LABEL: @vector_extract_slice_f16
func.func @vector_extract_slice_f16(%row: index) -> vector<[8]xf16> {
  // CHECK: arm_sme.extract_tile_slice {{.*}} : vector<[8]xf16> from vector<[8]x[8]xf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %slice = vector.extract %tile[%row] : vector<[8]xf16> from vector<[8]x[8]xf16>
  return %slice : vector<[8]xf16>
}

// -----

// CHECK-LABEL: @vector_extract_slice_bf16
func.func @vector_extract_slice_bf16(%row: index) -> vector<[8]xbf16> {
  // CHECK: arm_sme.extract_tile_slice {{.*}} : vector<[8]xbf16> from vector<[8]x[8]xbf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %slice = vector.extract %tile[%row] : vector<[8]xbf16> from vector<[8]x[8]xbf16>
  return %slice : vector<[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_extract_slice_f32
func.func @vector_extract_slice_f32(%row: index) -> vector<[4]xf32> {
  // CHECK: arm_sme.extract_tile_slice {{.*}} : vector<[4]xf32> from vector<[4]x[4]xf32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %slice = vector.extract %tile[%row] : vector<[4]xf32> from vector<[4]x[4]xf32>
  return %slice : vector<[4]xf32>
}

// -----

// CHECK-LABEL: @vector_extract_slice_f64
func.func @vector_extract_slice_f64(%row: index) -> vector<[2]xf64> {
  // CHECK: arm_sme.extract_tile_slice {{.*}} : vector<[2]xf64> from vector<[2]x[2]xf64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %slice = vector.extract %tile[%row] : vector<[2]xf64> from vector<[2]x[2]xf64>
  return %slice : vector<[2]xf64>
}

// -----

// CHECK-LABEL: @vector_extract_element(
// CHECK-SAME:                          %[[ROW:.*]]: index,
// CHECK-SAME:                          %[[COL:.*]]: index)
func.func @vector_extract_element(%row: index, %col: index) -> i32 {
  // CHECK-NEXT: %[[TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = arm_sme.extract_tile_slice %[[TILE]][%[[ROW]]] : vector<[4]xi32> from vector<[4]x[4]xi32>
  // CHECK-NEXT: %[[EL:.*]] = vector.extract %[[SLICE]]{{\[}}%[[COL]]] : i32 from vector<[4]xi32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %el = vector.extract %tile[%row, %col] : i32 from vector<[4]x[4]xi32>
  return %el : i32
}

// -----

// CHECK-LABEL: @vector_extract_element_i8
func.func @vector_extract_element_i8(%row: index, %col: index) -> i8 {
  // CHECK: %[[SLICE:.*]] = arm_sme.extract_tile_slice %{{.*}} : vector<[16]xi8> from vector<[16]x[16]xi8>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : i8 from vector<[16]xi8>
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %el = vector.extract %tile[%row, %col] : i8 from vector<[16]x[16]xi8>
  return %el : i8
}

// -----

// CHECK-LABEL: @vector_extract_element_i16
func.func @vector_extract_element_i16(%row: index, %col: index) -> i16 {
  // CHECK: %[[SLICE:.*]] = arm_sme.extract_tile_slice %{{.*}} : vector<[8]xi16> from vector<[8]x[8]xi16>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : i16 from vector<[8]xi16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %el = vector.extract %tile[%row, %col] : i16 from vector<[8]x[8]xi16>
  return %el : i16
}

// -----

// CHECK-LABEL: @vector_extract_element_i64
func.func @vector_extract_element_i64(%row: index, %col: index) -> i64 {
  // CHECK: %[[SLICE:.*]] = arm_sme.extract_tile_slice %{{.*}} : vector<[2]xi64> from vector<[2]x[2]xi64>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : i64 from vector<[2]xi64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %el = vector.extract %tile[%row, %col] : i64 from vector<[2]x[2]xi64>
  return %el : i64
}

// -----

// CHECK-LABEL: @vector_extract_element_i128
func.func @vector_extract_element_i128(%row: index, %col: index) -> i128 {
  // CHECK: %[[SLICE:.*]] = arm_sme.extract_tile_slice %{{.*}} : vector<[1]xi128> from vector<[1]x[1]xi128>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : i128 from vector<[1]xi128>
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %el = vector.extract %tile[%row, %col] : i128 from vector<[1]x[1]xi128>
  return %el : i128
}

// -----

// CHECK-LABEL: @vector_extract_element_f16
func.func @vector_extract_element_f16(%row: index, %col: index) -> f16 {
  // CHECK: %[[SLICE:.*]] = arm_sme.extract_tile_slice %{{.*}} : vector<[8]xf16> from vector<[8]x[8]xf16>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : f16 from vector<[8]xf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %el = vector.extract %tile[%row, %col] : f16 from vector<[8]x[8]xf16>
  return %el : f16
}

// -----

// CHECK-LABEL: @vector_extract_element_bf16
func.func @vector_extract_element_bf16(%row: index, %col: index) -> bf16 {
  // CHECK: %[[SLICE:.*]] = arm_sme.extract_tile_slice %{{.*}} : vector<[8]xbf16> from vector<[8]x[8]xbf16>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : bf16 from vector<[8]xbf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %el = vector.extract %tile[%row, %col] : bf16 from vector<[8]x[8]xbf16>
  return %el : bf16
}

// -----

// CHECK-LABEL: @vector_extract_element_f32
func.func @vector_extract_element_f32(%row: index, %col: index) -> f32 {
  // CHECK: %[[SLICE:.*]] = arm_sme.extract_tile_slice %{{.*}} : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : f32 from vector<[4]xf32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %el = vector.extract %tile[%row, %col] : f32 from vector<[4]x[4]xf32>
  return %el : f32
}

// -----

// CHECK-LABEL: @vector_extract_element_f64
func.func @vector_extract_element_f64(%row: index, %col: index) -> f64 {
  // CHECK: %[[SLICE:.*]] = arm_sme.extract_tile_slice %{{.*}} : vector<[2]xf64> from vector<[2]x[2]xf64>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : f64 from vector<[2]xf64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %el = vector.extract %tile[%row, %col] : f64 from vector<[2]x[2]xf64>
  return %el : f64
}

//===----------------------------------------------------------------------===//
// vector.extract --> arm_sve.psel
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @dynamic_vector_extract_mask_to_psel(
// CHECK-SAME:    %[[A:.*]]:  index, %[[B:.*]]: index, %[[INDEX:.*]]: index)
func.func @dynamic_vector_extract_mask_to_psel(%a: index, %b: index, %index: index) -> vector<[8]xi1>
{
  // CHECK: %[[MASK_ROWS:.*]] = vector.create_mask %[[A]] : vector<[4]xi1>
  // CHECK: %[[MASK_COLS:.*]] = vector.create_mask %[[B]] : vector<[8]xi1>
  // CHECK: arm_sve.psel %[[MASK_COLS]], %[[MASK_ROWS]][%[[INDEX]]] : vector<[8]xi1>, vector<[4]xi1>
  %mask = vector.create_mask %a, %b : vector<[4]x[8]xi1>
  %slice = vector.extract %mask[%index] : vector<[8]xi1> from vector<[4]x[8]xi1>
  return %slice : vector<[8]xi1>
}

// -----

// CHECK-LABEL: @vector_extract_mask_to_psel(
// CHECK-SAME:                               %[[A:.*]]: index,
// CHECK-SAME:                               %[[B:.*]]: index)
func.func @vector_extract_mask_to_psel(%a: index, %b: index) -> vector<[2]xi1>
{
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[MASK_ROWS:.*]] = vector.create_mask %[[A]] : vector<[16]xi1>
  // CHECK: %[[MASK_COLS:.*]] = vector.create_mask %[[B]] : vector<[2]xi1>
  // CHECK: arm_sve.psel %[[MASK_COLS]], %[[MASK_ROWS]][%[[C1]]] : vector<[2]xi1>, vector<[16]xi1>
  %mask = vector.create_mask %a, %b : vector<[16]x[2]xi1>
  %slice = vector.extract %mask[1] : vector<[2]xi1> from vector<[16]x[2]xi1>
  return %slice : vector<[2]xi1>
}
