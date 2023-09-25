// RUN: mlir-opt %s -convert-vector-to-arm-sme -split-input-file -allow-unregistered-dialect | FileCheck %s

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

// The following tests check the 'vector.transfer_write' -> 'arm_sme.intr.zero'
// lowering only occurs for vector types of correct rank, shape, element size
// and number of scalable dims.

// CHECK-LABEL: @transfer_write_2d_zero__bad_type
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__bad_type(%arg0 : memref<?x?xi4>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi4>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi4>, memref<?x?xi4>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__bad_shape
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d_zero__bad_shape(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[8]x[8]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__bad_rank
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d_zero__bad_rank(%arg0 : memref<?x?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<[16]x[16]x[16]xi8>, memref<?x?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__non_memref_type
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d_zero__non_memref_type(%arg0 : tensor<?x?xi8>) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi8>
  %0 = vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}

// -----

// CHECK-LABEL: @transfer_write_2d__fixed
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d__fixed(%vector : vector<16x16xi8>, %dest : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi8>, memref<?x?xi8>
  return
}

//===----------------------------------------------------------------------===//
// vector.broadcast
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:   func.func @broadcast_vec2d_from_i32(
// CHECK-SAME:                                        %[[SRC:.*]]: i32) {
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[SRC_1D:.*]] = vector.broadcast %[[SRC]] : i32 to vector<[4]xi32>
// CHECK: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i32
// CHECK: %[[TILE:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i32 to vector<[4]x[4]xi32>
// CHECK: %[[VSCALE:.*]] = vector.vscale
// CHECK: %[[NUM_TILE_SLICES:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
// CHECK: scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[NUM_TILE_SLICES]] step %[[C1]] {
// CHECK:   %[[C10:.*]] = arm_sme.move_vector_to_tile_slice %[[SRC_1D]], %[[TILE]], %[[TILE_SLICE_INDEX]] : vector<[4]xi32> into vector<[4]x[4]xi32>
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
// vector.transpose
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:   func.func @transpose_i8(
// CHECK-SAME:                            %[[TILE:.*]]: vector<[16]x[16]xi8>)
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[VSCALE:.*]] = vector.vscale
// CHECK:           %[[MIN_TILE_SLICES:.*]] = arith.muli %[[VSCALE]], %[[C16]] : index
// CHECK:           %[[NUM_TILE_SLICES:.*]] = memref.alloca(%[[MIN_TILE_SLICES]], %[[MIN_TILE_SLICES]]) : memref<?x?xi8>
// CHECK:           arm_sme.tile_store %[[TILE]], %[[NUM_TILE_SLICES]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi8>, vector<[16]x[16]xi8>
// CHECK:           arm_sme.tile_load %[[NUM_TILE_SLICES]]{{\[}}%[[C0]], %[[C0]]], <vertical> : memref<?x?xi8>, vector<[16]x[16]xi8>
func.func @transpose_i8(%arg0: vector<[16]x[16]xi8>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[16]x[16]xi8> to vector<[16]x[16]xi8>
  "prevent.dce"(%0) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_i16
// CHECK: arith.constant 8
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi16>, vector<[8]x[8]xi16>
// CHECK: arm_sme.tile_load {{.*}}, <vertical> : memref<?x?xi16>, vector<[8]x[8]xi16>
func.func @transpose_i16(%arg0: vector<[8]x[8]xi16>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[8]x[8]xi16> to vector<[8]x[8]xi16>
  "prevent.dce"(%0) : (vector<[8]x[8]xi16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_i32
// CHECK: arith.constant 4
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi32>, vector<[4]x[4]xi32>
// CHECK: arm_sme.tile_load {{.*}}, <vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
func.func @transpose_i32(%arg0: vector<[4]x[4]xi32>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[4]x[4]xi32> to vector<[4]x[4]xi32>
  "prevent.dce"(%0) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_i64
// CHECK: arith.constant 2
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi64>, vector<[2]x[2]xi64>
// CHECK: arm_sme.tile_load {{.*}}, <vertical> : memref<?x?xi64>, vector<[2]x[2]xi64>
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
// CHECK: arm_sme.tile_load {{.*}}, <vertical> : memref<?x?xi128>, vector<[1]x[1]xi128>
func.func @transpose_i128(%arg0: vector<[1]x[1]xi128>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[1]x[1]xi128> to vector<[1]x[1]xi128>
  "prevent.dce"(%0) : (vector<[1]x[1]xi128>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_f16
// CHECK: arith.constant 8
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf16>, vector<[8]x[8]xf16>
// CHECK: arm_sme.tile_load {{.*}}, <vertical> : memref<?x?xf16>, vector<[8]x[8]xf16>
func.func @transpose_f16(%arg0: vector<[8]x[8]xf16>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[8]x[8]xf16> to vector<[8]x[8]xf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_bf16
// CHECK: arith.constant 8
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xbf16>, vector<[8]x[8]xbf16>
// CHECK: arm_sme.tile_load {{.*}}, <vertical> : memref<?x?xbf16>, vector<[8]x[8]xbf16>
func.func @transpose_bf16(%arg0: vector<[8]x[8]xbf16>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[8]x[8]xbf16> to vector<[8]x[8]xbf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xbf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_f32
// CHECK: arith.constant 4
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK: arm_sme.tile_load {{.*}}, <vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @transpose_f32(%arg0: vector<[4]x[4]xf32>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[4]x[4]xf32> to vector<[4]x[4]xf32>
  "prevent.dce"(%0) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: @transpose_f64
// CHECK: arith.constant 2
// CHECK: arm_sme.tile_store {{.*}} : memref<?x?xf64>, vector<[2]x[2]xf64>
// CHECK: arm_sme.tile_load {{.*}}, <vertical> : memref<?x?xf64>, vector<[2]x[2]xf64>
func.func @transpose_f64(%arg0: vector<[2]x[2]xf64>) {
  %0 = vector.transpose %arg0, [1, 0] : vector<[2]x[2]xf64> to vector<[2]x[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
  return
}
