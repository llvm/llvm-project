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
// CHECK:   %[[NEW_TILE:.*]] = arm_sme.move_vector_to_tile_slice %[[SRC_1D]], %[[CURRENT_TILE]], %[[TILE_SLICE_INDEX]] : vector<[4]xi32> into vector<[4]x[4]xi32>
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
// CHECK:   arm_sme.move_vector_to_tile_slice %[[SRC_1D]], {{.*}}
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
// CHECK:   arm_sme.move_vector_to_tile_slice %[[SRC]], {{.*}}
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
// CHECK:     arm_sme.move_vector_to_tile_slice %[[BCST]], {{.*}} : vector<[4]xi32> into vector<[4]x[4]xi32>
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
// CHECK:     arm_sme.move_vector_to_tile_slice %[[BCST]], {{.*}} : vector<[8]xf16> into vector<[8]x[8]xf16>
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
// CHECK-NEXT:        %[[TILE_SLICE:.*]] = arm_sme.move_tile_slice_to_vector %[[TILE]][%[[TILE_SLICE_INDEX]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
// CHECK-NEXT:        vector.print %[[TILE_SLICE]] : vector<[4]xf32>
