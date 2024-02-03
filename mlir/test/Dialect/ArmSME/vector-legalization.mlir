// RUN: mlir-opt %s -arm-sme-vector-legalization -cse -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: @outerproduct_f32_scalable_8x8_no_acc(
// CHECK-SAME:                                        %[[LHS:.*]]: vector<[8]xf32>,
// CHECK-SAME:                                        %[[RHS:.*]]: vector<[8]xf32>)
// CHECK-SAME: -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>)
func.func @outerproduct_f32_scalable_8x8_no_acc(%lhs: vector<[8]xf32>, %rhs: vector<[8]xf32>) -> vector<[8]x[8]xf32>
{
  // CHECK-DAG: %[[LHS_0:.*]] = vector.scalable.extract %[[LHS]][0] : vector<[4]xf32> from vector<[8]xf32>
  // CHECK-DAG: %[[RHS_0:.*]] = vector.scalable.extract %[[RHS]][0] : vector<[4]xf32> from vector<[8]xf32>
  // CHECK-DAG: %[[LHS_1:.*]] = vector.scalable.extract %[[LHS]][4] : vector<[4]xf32> from vector<[8]xf32>
  // CHECK-DAG: %[[RHS_1:.*]] = vector.scalable.extract %[[RHS]][4] : vector<[4]xf32> from vector<[8]xf32>
  // CHECK-DAG: %[[TOP_LEFT:.*]] = vector.outerproduct %[[LHS_0]], %[[RHS_0]] : vector<[4]xf32>, vector<[4]xf32>
  // CHECK-DAG: %[[TOP_RIGHT:.*]] = vector.outerproduct %[[LHS_0]], %[[RHS_1]] : vector<[4]xf32>, vector<[4]xf32>
  // CHECK-DAG: %[[BOTTOM_LEFT:.*]] = vector.outerproduct %[[LHS_1]], %[[RHS_0]] : vector<[4]xf32>, vector<[4]xf32>
  // CHECK-DAG: %[[BOTTOM_RIGHT:.*]] = vector.outerproduct %[[LHS_1]], %[[RHS_1]] : vector<[4]xf32>, vector<[4]xf32>
  // CHECK-NEXT: return %[[TOP_LEFT]], %[[TOP_RIGHT]], %[[BOTTOM_LEFT]], %[[BOTTOM_RIGHT]] : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
  %0 = vector.outerproduct %lhs, %rhs : vector<[8]xf32>, vector<[8]xf32>
  return %0 : vector<[8]x[8]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_f32_scalable_4x16_acc(
// CHECK-SAME:                                      %[[LHS:.*]]: vector<[4]xf32>,
// CHECK-SAME:                                      %[[RHS:.*]]: vector<[16]xf32>,
// CHECK-SAME:                                      %[[ACC_0:[A-Za-z0-9]*]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                      %[[ACC_1:[A-Za-z0-9]*]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                      %[[ACC_2:[A-Za-z0-9]*]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                      %[[ACC_3:[A-Za-z0-9]*]]: vector<[4]x[4]xf32>)
// CHECK-SAME: -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>)
func.func @outerproduct_f32_scalable_4x16_acc(%lhs: vector<[4]xf32>, %rhs: vector<[16]xf32>, %acc: vector<[4]x[16]xf32>) -> vector<[4]x[16]xf32>
{
  // CHECK-DAG: %[[LHS_0:.*]] = vector.scalable.extract %[[LHS]][0] : vector<[4]xf32> from vector<[4]xf32>
  // CHECK-DAG: %[[RHS_0:.*]] = vector.scalable.extract %[[RHS]][0] : vector<[4]xf32> from vector<[16]xf32>
  // CHECK-DAG: %[[RHS_1:.*]] = vector.scalable.extract %[[RHS]][4] : vector<[4]xf32> from vector<[16]xf32>
  // CHECK-DAG: %[[RHS_2:.*]] = vector.scalable.extract %[[RHS]][8] : vector<[4]xf32> from vector<[16]xf32>
  // CHECK-DAG: %[[RHS_3:.*]] = vector.scalable.extract %[[RHS]][12] : vector<[4]xf32> from vector<[16]xf32>
  // CHECK-DAG: %[[RES_0:.*]] = vector.outerproduct %[[LHS_0]], %[[RHS_0]], %[[ACC_0]] {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32>
  // CHECK-DAG: %[[RES_1:.*]] = vector.outerproduct %[[LHS_0]], %[[RHS_1]], %[[ACC_1]] {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32>
  // CHECK-DAG: %[[RES_2:.*]] = vector.outerproduct %[[LHS_0]], %[[RHS_2]], %[[ACC_2]] {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32>
  // CHECK-DAG: %[[RES_3:.*]] = vector.outerproduct %[[LHS_0]], %[[RHS_3]], %[[ACC_3]] {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32>
  // CHECK-NEXT: return %[[RES_0]], %[[RES_1]], %[[RES_2]], %[[RES_3]] : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
  %0 = vector.outerproduct %lhs, %rhs, %acc : vector<[4]xf32>, vector<[16]xf32>
  return %0 : vector<[4]x[16]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_f32_masked_scalable_16x4(
// CHECK-SAME:                                         %[[LHS:.*]]: vector<[16]xf32>,
// CHECK-SAME:                                         %[[RHS:.*]]: vector<[4]xf32>,
// CHECK-SAME:                                         %[[LHS_DIM:.*]]: index,
// CHECK-SAME:                                         %[[RHS_DIM:.*]]: index)
// CHECK-SAME: -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>)
func.func @outerproduct_f32_masked_scalable_16x4(%lhs: vector<[16]xf32>, %rhs: vector<[4]xf32>, %lhs_dim: index, %rhs_dim: index) -> vector<[16]x[4]xf32>
{
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[MINUS_4:.*]] = arith.constant -4 : index
  // CHECK-DAG: %[[MINUS_8:.*]] = arith.constant -8 : index
  // CHECK-DAG: %[[MINUS_12:.*]] = arith.constant -12 : index
  // CHECK-DAG: %[[MINUS_4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[MINUS_4]] : index
  // CHECK-DAG: %[[MINUS_8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[MINUS_8]] : index
  // CHECK-DAG: %[[MINUS_12_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[MINUS_12]] : index
  // CHECK-DAG: %[[LHS_0:.*]] = vector.scalable.extract %[[LHS]][0] : vector<[4]xf32> from vector<[16]xf32>
  // CHECK-DAG: %[[LHS_1:.*]] = vector.scalable.extract %[[LHS]][4] : vector<[4]xf32> from vector<[16]xf32>
  // CHECK-DAG: %[[LHS_2:.*]] = vector.scalable.extract %[[LHS]][8] : vector<[4]xf32> from vector<[16]xf32>
  // CHECK-DAG: %[[LHS_3:.*]] = vector.scalable.extract %[[LHS]][12] : vector<[4]xf32> from vector<[16]xf32>
  // CHECK-DAG: %[[RHS_0:.*]] = vector.scalable.extract %[[RHS]][0] : vector<[4]xf32> from vector<[4]xf32>
  // CHECK-DAG: %[[MASK_0:.*]] = vector.create_mask %[[LHS_DIM]], %[[RHS_DIM]] : vector<[4]x[4]xi1>
  // CHECK-DAG: %[[TILE_1_LHS_DIM:.*]] = arith.addi %[[LHS_DIM]], %[[MINUS_4_VSCALE]] : index
  // CHECK-DAG: %[[MASK_1:.*]] = vector.create_mask %[[TILE_1_LHS_DIM]], %[[RHS_DIM]] : vector<[4]x[4]xi1>
  // CHECK-DAG: %[[TILE_2_LHS_DIM:.*]] = arith.addi %[[LHS_DIM]], %[[MINUS_8_VSCALE]] : index
  // CHECK-DAG: %[[MASK_2:.*]] = vector.create_mask %[[TILE_2_LHS_DIM]], %[[RHS_DIM]] : vector<[4]x[4]xi1>
  // CHECK-DAG: %[[TILE_3_LHS_DIM:.*]] = arith.addi %[[LHS_DIM]], %[[MINUS_12_VSCALE]] : index
  // CHECK-DAG: %[[MASK_3:.*]] = vector.create_mask %[[TILE_3_LHS_DIM]], %[[RHS_DIM]] : vector<[4]x[4]xi1>
  // CHECK-DAG: %[[RES_0:.*]] = vector.mask %[[MASK_0]] { vector.outerproduct %[[LHS_0]], %[[RHS_0]] : vector<[4]xf32>, vector<[4]xf32> } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>
  // CHECK-DAG: %[[RES_1:.*]] = vector.mask %[[MASK_1]] { vector.outerproduct %[[LHS_1]], %[[RHS_0]] : vector<[4]xf32>, vector<[4]xf32> } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>
  // CHECK-DAG: %[[RES_2:.*]] = vector.mask %[[MASK_2]] { vector.outerproduct %[[LHS_2]], %[[RHS_0]] : vector<[4]xf32>, vector<[4]xf32> } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>
  // CHECK-DAG: %[[RES_3:.*]] = vector.mask %[[MASK_3]] { vector.outerproduct %[[LHS_3]], %[[RHS_0]] : vector<[4]xf32>, vector<[4]xf32> } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>
  // CHECK-NEXT: return %[[RES_0]], %[[RES_1]], %[[RES_2]], %[[RES_3]] : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
  %mask = vector.create_mask %lhs_dim, %rhs_dim : vector<[16]x[4]xi1>
  %0 = vector.mask %mask { vector.outerproduct %lhs, %rhs : vector<[16]xf32>, vector<[4]xf32> } : vector<[16]x[4]xi1> -> vector<[16]x[4]xf32>
  return %0 : vector<[16]x[4]xf32>
}

// -----

/// This demonstrates a rectangular tiling that uses all f64 accumulators.

// CHECK-LABEL: @outerproduct_f64_scalable_8x4_no_acc(
// CHECK-SAME:                                        %[[LHS:.*]]: vector<[8]xf64>,
// CHECK-SAME:                                        %[[RHS:.*]]: vector<[4]xf64>)
// CHECK-SAME: -> (vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>)
func.func @outerproduct_f64_scalable_8x4_no_acc(%lhs: vector<[8]xf64>, %rhs: vector<[4]xf64>) -> vector<[8]x[4]xf64>
{
  // CHECK-DAG: %[[LHS_0:.*]] = vector.scalable.extract %[[LHS]][0] : vector<[2]xf64> from vector<[8]xf64>
  // CHECK-DAG: %[[LHS_1:.*]] = vector.scalable.extract %[[LHS]][2] : vector<[2]xf64> from vector<[8]xf64>
  // CHECK-DAG: %[[LHS_2:.*]] = vector.scalable.extract %[[LHS]][4] : vector<[2]xf64> from vector<[8]xf64>
  // CHECK-DAG: %[[LHS_3:.*]] = vector.scalable.extract %[[LHS]][6] : vector<[2]xf64> from vector<[8]xf64>
  // CHECK-DAG: %[[RHS_0:.*]] = vector.scalable.extract %[[RHS]][0] : vector<[2]xf64> from vector<[4]xf64>
  // CHECK-DAG: %[[RHS_1:.*]] = vector.scalable.extract %[[RHS]][2] : vector<[2]xf64> from vector<[4]xf64>
  // CHECK-DAG: %[[RES_0:.*]] = vector.outerproduct %[[LHS_0]], %[[RHS_0]] : vector<[2]xf64>, vector<[2]xf64>
  // CHECK-DAG: %[[RES_1:.*]] = vector.outerproduct %[[LHS_0]], %[[RHS_1]] : vector<[2]xf64>, vector<[2]xf64>
  // CHECK-DAG: %[[RES_2:.*]] = vector.outerproduct %[[LHS_1]], %[[RHS_0]] : vector<[2]xf64>, vector<[2]xf64>
  // CHECK-DAG: %[[RES_3:.*]] = vector.outerproduct %[[LHS_1]], %[[RHS_1]] : vector<[2]xf64>, vector<[2]xf64>
  // CHECK-DAG: %[[RES_4:.*]] = vector.outerproduct %[[LHS_2]], %[[RHS_0]] : vector<[2]xf64>, vector<[2]xf64>
  // CHECK-DAG: %[[RES_5:.*]] = vector.outerproduct %[[LHS_2]], %[[RHS_1]] : vector<[2]xf64>, vector<[2]xf64>
  // CHECK-DAG: %[[RES_6:.*]] = vector.outerproduct %[[LHS_3]], %[[RHS_0]] : vector<[2]xf64>, vector<[2]xf64>
  // CHECK-DAG: %[[RES_7:.*]] = vector.outerproduct %[[LHS_3]], %[[RHS_1]] : vector<[2]xf64>, vector<[2]xf64>
  // CHECK-NEXT: return %[[RES_0]], %[[RES_1]], %[[RES_2]], %[[RES_3]], %[[RES_4]], %[[RES_5]], %[[RES_6]], %[[RES_7]] : vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>, vector<[2]x[2]xf64>
  %0 = vector.outerproduct %lhs, %rhs : vector<[8]xf64>, vector<[4]xf64>
  return %0 : vector<[8]x[4]xf64>
}

// -----

// CHECK-LABEL: @transfer_read_f32_scalable_8x8(
// CHECK-SAME:                                  %[[SRC:.*]]: memref<?x?xi32>)
// CHECK-SAME: -> (vector<[4]x[4]xi32>, vector<[4]x[4]xi32>, vector<[4]x[4]xi32>, vector<[4]x[4]xi32>)
func.func @transfer_read_f32_scalable_8x8(%src: memref<?x?xi32>) -> vector<[8]x[8]xi32>
{
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  // CHECK-DAG: %[[TOP_LEFT:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C0]]], %[[C0_I32]] {in_bounds = [true, true]} : memref<?x?xi32>, vector<[4]x[4]xi32>
  // CHECK-DAG: %[[TOP_RIGHT:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C4_VSCALE]]], %[[C0_I32]] {in_bounds = [true, true]} : memref<?x?xi32>, vector<[4]x[4]xi32>
  // CHECK-DAG: %[[BOTTOM_LEFT:.*]] = vector.transfer_read %[[SRC]][%[[C4_VSCALE]], %[[C0]]], %[[C0_I32]] {in_bounds = [true, true]} : memref<?x?xi32>, vector<[4]x[4]xi32>
  // CHECK-DAG: %[[BOTTOM_RIGHT:.*]] = vector.transfer_read %[[SRC]][%[[C4_VSCALE]], %[[C4_VSCALE]]], %[[C0_I32]] {in_bounds = [true, true]} : memref<?x?xi32>, vector<[4]x[4]xi32>
  // CHECK-NEXT: return %[[TOP_LEFT]], %[[TOP_RIGHT]], %[[BOTTOM_LEFT]], %[[BOTTOM_RIGHT]] : vector<[4]x[4]xi32>, vector<[4]x[4]xi32>, vector<[4]x[4]xi32>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i32
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xi32>, vector<[8]x[8]xi32>
  return %0 : vector<[8]x[8]xi32>
}

// -----

// CHECK-LABEL: @transfer_read_i16_scalable_8x16_masked(
// CHECK-SAME:                                          %[[SRC:.*]]: memref<?x?xi16>,
// CHECK-SAME:                                          %[[DIM0:.*]]: index,
// CHECK-SAME:                                          %[[DIM1:.*]]: index)
// CHECK-SAME: -> (vector<[8]x[8]xi16>, vector<[8]x[8]xi16>)
func.func @transfer_read_i16_scalable_8x16_masked(%src: memref<?x?xi16>, %dim0: index, %dim1: index) -> vector<[8]x[16]xi16>
{
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[MINUS_8:.*]] = arith.constant -8 : index
  // CHECK-DAG: %[[C0_I16:.*]] = arith.constant 0 : i16
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[MINUS_8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[MINUS_8]] : index
  // CHECK-DAG: %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  // CHECK-DAG: %[[RIGHT_DIM_1:.*]] = arith.addi %[[DIM1]], %[[MINUS_8_VSCALE]] : index
  // CHECK-DAG: %[[LEFT_MASK:.*]] = vector.create_mask %[[DIM0]], %[[DIM1]] : vector<[8]x[8]xi1>
  // CHECK-DAG: %[[RIGHT_MASK:.*]] = vector.create_mask %[[DIM0]], %[[RIGHT_DIM_1]] : vector<[8]x[8]xi1>
  // CHECK-DAG: %[[LEFT:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C0]]], %[[C0_I16]], %[[LEFT_MASK]] {in_bounds = [true, true]} : memref<?x?xi16>, vector<[8]x[8]xi16>
  // CHECK-DAG: %[[RIGHT:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C8_VSCALE]]], %[[C0_I16]], %[[RIGHT_MASK]] {in_bounds = [true, true]} : memref<?x?xi16>, vector<[8]x[8]xi16>
  // CHECK-NEXT: return %[[LEFT]], %[[RIGHT]] : vector<[8]x[8]xi16>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i16
  %mask = vector.create_mask %dim0, %dim1 : vector<[8]x[16]xi1>
  %0 = vector.transfer_read %src[%c0, %c0], %pad, %mask {in_bounds = [true, true]} : memref<?x?xi16>, vector<[8]x[16]xi16>
  return %0 : vector<[8]x[16]xi16>
}

// -----

// CHECK-LABEL: @transfer_write_f16_scalable_16x8(
// CHECK-SAME:                                    %[[DEST:.*]]: memref<?x?xf16>,
// CHECK-SAME:                                    %[[TOP:.*]]: vector<[8]x[8]xf16>,
// CHECK-SAME:                                    %[[BOTTOM:.*]]: vector<[8]x[8]xf16>)
func.func @transfer_write_f16_scalable_16x8(%dest: memref<?x?xf16>, %vec: vector<[16]x[8]xf16>)
{
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  // CHECK-DAG: vector.transfer_write %[[TOP]], %[[DEST]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<[8]x[8]xf16>, memref<?x?xf16>
  // CHECK-DAG: vector.transfer_write %[[BOTTOM]], %[[DEST]][%[[C8_VSCALE]], %[[C0]]] {in_bounds = [true, true]} : vector<[8]x[8]xf16>, memref<?x?xf16>
  // CHECK-NEXT: return
  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[8]xf16>, memref<?x?xf16>
  return
}

// -----

/// This is already a legal type. It should be ignored.

// CHECK-LABEL: @transfer_write_i8_scalable_16x16_masked
func.func @transfer_write_i8_scalable_16x16_masked(%dest: memref<?x?xi8>, %vec: vector<[16]x[16]xi8>, %dim0: index, %dim1: index)
{
  // CHECK: vector.transfer_write {{.*}} : vector<[16]x[16]xi8>, memref<?x?xi8>
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %dim0, %dim0 : vector<[16]x[16]xi1>
  vector.transfer_write %vec, %dest[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

// -----

#transpose = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @transpose_f32_scalable_4x16_via_read(
// CHECK-SAME:                                        %[[SRC:.*]]: memref<?x?xf32>,
// CHECK-SAME:                                        %[[DEST:.*]]: memref<?x?xf32>)
func.func @transpose_f32_scalable_4x16_via_read(%src: memref<?x?xf32>, %dest: memref<?x?xf32>)
{
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
  // CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  // CHECK-DAG: %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  // CHECK-DAG: %[[C12_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C12]] : index
  // CHECK-DAG: %[[TILE_0:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C0]]], %[[PAD]] {in_bounds = [true, true], permutation_map = #{{.*}}} : memref<?x?xf32>, vector<[4]x[4]xf32>
  // CHECK-DAG: %[[TILE_1:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C4_VSCALE]]], %[[PAD]] {in_bounds = [true, true], permutation_map = #{{.*}}} : memref<?x?xf32>, vector<[4]x[4]xf32>
  // CHECK-DAG: %[[TILE_2:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C8_VSCALE]]], %[[PAD]] {in_bounds = [true, true], permutation_map = #{{.*}}} : memref<?x?xf32>, vector<[4]x[4]xf32>
  // CHECK-DAG: %[[TILE_3:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C12_VSCALE]]], %[[PAD]] {in_bounds = [true, true], permutation_map = #{{.*}}} : memref<?x?xf32>, vector<[4]x[4]xf32>
  // CHECK-DAG: vector.transfer_write %[[TILE_0]], %[[DEST]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // CHECK-DAG: vector.transfer_write %[[TILE_1]], %[[DEST]][%[[C4_VSCALE]], %[[C0]]] {in_bounds = [true, true]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // CHECK-DAG: vector.transfer_write %[[TILE_2]], %[[DEST]][%[[C8_VSCALE]], %[[C0]]] {in_bounds = [true, true]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // CHECK-DAG: vector.transfer_write %[[TILE_3]], %[[DEST]][%[[C12_VSCALE]], %[[C0]]] {in_bounds = [true, true]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT: return
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %src[%c0, %c0], %pad {permutation_map = #transpose, in_bounds = [true, true]} : memref<?x?xf32>, vector<[16]x[4]xf32>
  vector.transfer_write %0, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[4]xf32>, memref<?x?xf32>
  return
}

// -----

#transpose = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @transpose_f32_scalable_4x16_via_write(
// CHECK-SAME:                                         %[[SRC:.*]]: memref<?x?xf32>,
// CHECK-SAME:                                         %[[DEST:.*]]: memref<?x?xf32>)
func.func @transpose_f32_scalable_4x16_via_write(%src: memref<?x?xf32>, %dest: memref<?x?xf32>)
{
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
  // CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  // CHECK-DAG: %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  // CHECK-DAG: %[[C12_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C12]] : index
  // CHECK-DAG: %[[TILE_0:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C0]]], %[[PAD]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>
  // CHECK-DAG: %[[TILE_1:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C4_VSCALE]]], %[[PAD]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>
  // CHECK-DAG: %[[TILE_2:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C8_VSCALE]]], %[[PAD]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>
  // CHECK-DAG: %[[TILE_3:.*]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C12_VSCALE]]], %[[PAD]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>
  // CHECK-DAG: vector.transfer_write %[[TILE_0]], %[[DEST]][%[[C0]], %[[C0]]] {in_bounds = [true, true], permutation_map = #{{.*}}} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // CHECK-DAG: vector.transfer_write %[[TILE_1]], %[[DEST]][%[[C4_VSCALE]], %[[C0]]] {in_bounds = [true, true], permutation_map = #{{.*}}} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // CHECK-DAG: vector.transfer_write %[[TILE_2]], %[[DEST]][%[[C8_VSCALE]], %[[C0]]] {in_bounds = [true, true], permutation_map = #{{.*}}} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // CHECK-DAG: vector.transfer_write %[[TILE_3]], %[[DEST]][%[[C12_VSCALE]], %[[C0]]] {in_bounds = [true, true], permutation_map = #{{.*}}} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT: return
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[16]xf32>
  vector.transfer_write %0, %dest[%c0, %c0] {permutation_map = #transpose, in_bounds = [true, true]} : vector<[4]x[16]xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @extract_from_vector_create_mask_non_constant_dim(
// CHECK-SAME:                                                    %[[DIM0:[a-z0-9]+]]: index,
// CHECK-SAME:                                                    %[[DIM1:[a-z0-9]+]]: index,
// CHECK-SAME:                                                    %[[DIM2:[a-z0-9]+]]: index)
func.func @extract_from_vector_create_mask_non_constant_dim(%dim0: index, %dim1: index, %dim2: index) -> vector<[4]x[4]xi1> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[DIM0_CMP:.*]] = arith.cmpi sgt, %[[DIM0]], %[[C2]] : index
  // CHECK-NEXT: %[[NEW_DIM0:.*]] = arith.select %[[DIM0_CMP]], %[[DIM1]], %[[C0]] : index
  // CHECK-NEXT: %[[EXTRACT:.*]] = vector.create_mask %[[NEW_DIM0]], %[[DIM2]] : vector<[4]x[4]xi1>
  // CHECK-NEXT: return %[[EXTRACT]]
  %mask = vector.create_mask %dim0, %dim1, %dim2 : vector<4x[4]x[4]xi1>
  %extract = vector.extract %mask[2] : vector<[4]x[4]xi1> from vector<4x[4]x[4]xi1>
  return %extract : vector<[4]x[4]xi1>
}

// -----

// CHECK-LABEL: @non_constant_extract_from_vector_create_mask_non_constant(
// CHECK-SAME:                                                             %[[INDEX:[a-z0-9]+]]: index,
// CHECK-SAME:                                                             %[[DIM0:[a-z0-9]+]]: index,
// CHECK-SAME:                                                             %[[DIM1:[a-z0-9]+]]: index,
// CHECK-SAME:                                                             %[[DIM2:[a-z0-9]+]]: index)
func.func @non_constant_extract_from_vector_create_mask_non_constant(%index: index, %dim0: index, %dim1: index, %dim2: index) -> vector<[4]x[4]xi1> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[DIM0_CMP:.*]] = arith.cmpi slt, %[[INDEX]], %[[DIM0]] : index
  // CHECK-NEXT: %[[NEW_DIM0:.*]] = arith.select %[[DIM0_CMP]], %[[DIM1]], %[[C0]] : index
  // CHECK-NEXT: %[[EXTRACT:.*]] = vector.create_mask %[[NEW_DIM0]], %[[DIM2]] : vector<[4]x[4]xi1>
  // CHECK-NEXT: return %[[EXTRACT]]
  %mask = vector.create_mask %dim0, %dim1, %dim2 : vector<4x[4]x[4]xi1>
  %extract = vector.extract %mask[%index] : vector<[4]x[4]xi1> from vector<4x[4]x[4]xi1>
  return %extract : vector<[4]x[4]xi1>
}
