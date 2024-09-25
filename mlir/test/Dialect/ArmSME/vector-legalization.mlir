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
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  // CHECK-NEXT: scf.for %[[I:.*]] = %[[C0]] to %[[C8_VSCALE]] step %[[C1]] {
  // CHECK-NEXT:   %[[TOP_SLICE:.*]] = vector.extract %[[TOP]][%[[I]]] : vector<[8]xf16> from vector<[8]x[8]xf16>
  // CHECK-NEXT:   vector.transfer_write %[[TOP_SLICE]], %[[DEST]][%[[I]], %[[C0]]] {in_bounds = [true]} : vector<[8]xf16>, memref<?x?xf16>
  // CHECK-NEXT:   %[[BOTTOM_I:.*]] = arith.addi %[[C8_VSCALE]], %[[I]] : index
  // CHECK-NEXT:   %[[BOTTOM_SLICE:.*]] = vector.extract %[[BOTTOM]][%[[I]]] : vector<[8]xf16> from vector<[8]x[8]xf16>
  // CHECK-NEXT:   vector.transfer_write %[[BOTTOM_SLICE]], %[[DEST]][%[[BOTTOM_I]], %[[C0]]] {in_bounds = [true]} : vector<[8]xf16>, memref<?x?xf16>
  // CHECK-NEXT: }
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

// CHECK-LABEL: @transfer_write_f32_scalable_8x8_masked(
// CHECK-SAME:                                    %[[DEST:[a-z0-9]+]]: memref<?x?xf32>,
// CHECK-SAME:                                    %[[DIM_0:[a-z0-9]+]]: index,
// CHECK-SAME:                                    %[[DIM_1:[a-z0-9]+]]: index,
// CHECK-SAME:                                    %[[TILE_0:[a-z0-9]+]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                    %[[TILE_1:[a-z0-9]+]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                    %[[TILE_2:[a-z0-9]+]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                    %[[TILE_3:[a-z0-9]+]]: vector<[4]x[4]xf32>)
func.func @transfer_write_f32_scalable_8x8_masked(%dest: memref<?x?xf32>, %dim0: index, %dim1: index, %vec: vector<[8]x[8]xf32>)
{
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  // CHECK-DAG: %[[MASK:.*]] =  vector.create_mask %[[DIM_0]], %[[DIM_1]] : vector<[8]x[8]xi1>
  // CHECK-NEXT: scf.for %[[I:.*]] = %[[C0]] to %[[C4_VSCALE]] step %[[C1]] {
  // CHECK-NEXT:   %[[UPPER_SLICE_MASK:.*]] = vector.extract %[[MASK]][%[[I]]] : vector<[8]xi1> from vector<[8]x[8]xi1>
  // CHECK-NEXT:   %[[TILE_0_SLICE_MASK:.*]] = vector.scalable.extract %[[UPPER_SLICE_MASK]][0] : vector<[4]xi1> from vector<[8]xi1>
  // CHECK-NEXT:   %[[TILE_0_SLICE:.*]] = vector.extract %[[TILE_0]][%[[I]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT:   vector.transfer_write %[[TILE_0_SLICE]], %[[DEST]][%[[I]], %[[C0]]], %[[TILE_0_SLICE_MASK]] {in_bounds = [true]} : vector<[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT:   %[[TILE_1_SLICE_MASK:.*]] = vector.scalable.extract %[[UPPER_SLICE_MASK]][4] : vector<[4]xi1> from vector<[8]xi1>
  // CHECK-NEXT:   %[[TILE_1_SLICE:.*]] = vector.extract %[[TILE_1]][%[[I]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT:   vector.transfer_write %[[TILE_1_SLICE]], %[[DEST]][%[[I]], %[[C4_VSCALE]]], %[[TILE_1_SLICE_MASK]] {in_bounds = [true]} : vector<[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT:   %[[LOWER_SLICE_I:.*]] = arith.addi %[[C4_VSCALE]], %[[I]] : index
  // CHECK-NEXT:   %[[LOWER_SLICE_MASK:.*]] = vector.extract %[[MASK]][%[[LOWER_SLICE_I]]] : vector<[8]xi1> from vector<[8]x[8]xi1>
  // CHECK-NEXT:   %[[TILE_2_SLICE_MASK:.*]] = vector.scalable.extract %[[LOWER_SLICE_MASK]][0] : vector<[4]xi1> from vector<[8]xi1>
  // CHECK-NEXT:   %[[TILE_2_SLICE:.*]] = vector.extract %[[TILE_2]][%[[I]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT:   vector.transfer_write %[[TILE_2_SLICE]], %[[DEST]][%[[LOWER_SLICE_I]], %[[C0]]], %[[TILE_2_SLICE_MASK]] {in_bounds = [true]} : vector<[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT:   %[[TILE_3_SLICE_MASK:.*]] = vector.scalable.extract %[[LOWER_SLICE_MASK]][4] : vector<[4]xi1> from vector<[8]xi1>
  // CHECK-NEXT:   %[[TILE_3_SLICE:.*]] = vector.extract %[[TILE_3]][%[[I]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT:   vector.transfer_write %[[TILE_3_SLICE:.*]], %[[DEST]][%[[LOWER_SLICE_I]], %[[C4_VSCALE]]], %[[TILE_3_SLICE_MASK]] {in_bounds = [true]} : vector<[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT: }
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %dim0, %dim1 : vector<[8]x[8]xi1>
  vector.transfer_write %vec, %dest[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  return
}

// -----

// Tensor semantics are not supported for the store loop lowering.

// CHECK-LABEL: @negative_transfer_write_f32_scalable_8x8_tensor
// CHECK-NOT: scf.for
func.func @negative_transfer_write_f32_scalable_8x8_tensor(%dest: tensor<?x?xf32>, %vec: vector<[8]x[8]xf32>)
{
  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xf32>, tensor<?x?xf32>
  return
}

// -----

#transpose = affine_map<(d0, d1) -> (d1, d0)>

// Transposes are not supported for the store loop lowering.

// CHECK-LABEL: @negative_transfer_write_f32_scalable_8x8_tensor
// CHECK-NOT: scf.for
func.func @negative_transfer_write_f32_scalable_8x8_tensor(%dest: tensor<?x?xf32>, %dim0: index, %dim1: index, %vec: vector<[8]x[8]xf32>)
{
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %dim0, %dim1 : vector<[8]x[8]xi1>
  vector.transfer_write %vec, %dest[%c0, %c0], %mask {permutation_map = #transpose, in_bounds = [true, true]} : vector<[8]x[8]xf32>, tensor<?x?xf32>
  return
}

// -----

// Masked writes where any dimension of the mask is > 16 are not supported for the store loop lowering.

// CHECK-LABEL: @negative_transfer_write_f32_scalable_32x32
// CHECK-NOT: scf.for
func.func @negative_transfer_write_f32_scalable_32x32(%dest: memref<?x?xf32>, %dim0: index, %dim1: index, %vec: vector<[32]x[32]xf32>)
{
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %dim0, %dim1 : vector<[32]x[32]xi1>
  vector.transfer_write %vec, %dest[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[32]x[32]xf32>, memref<?x?xf32>
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
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
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
  // CHECK-NEXT: scf.for %[[I:.*]] = %[[C0]] to %[[C4_VSCALE]] step %[[C1]] {
  // CHECK-NEXT:   %[[TILE_0_SLICE:.*]] = vector.extract %[[TILE_0]][%[[I]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT:   vector.transfer_write %[[TILE_0_SLICE]], %[[DEST]][%[[I]], %[[C0]]] {in_bounds = [true]} : vector<[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT:   %[[TILE_1_I:.*]] = arith.addi %[[C4_VSCALE]], %[[I]] : index
  // CHECK-NEXT:   %[[TILE_1_SLICE:.*]] = vector.extract %[[TILE_1]][%[[I]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT:   vector.transfer_write %[[TILE_1_SLICE]], %[[DEST]][%[[TILE_1_I]], %[[C0]]] {in_bounds = [true]} : vector<[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT:   %[[TILE_2_I:.*]] = arith.addi %[[C8_VSCALE]], %[[I]] : index
  // CHECK-NEXT:   %[[TILE_2_SLICE:.*]] = vector.extract %[[TILE_2]][%[[I]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT:   vector.transfer_write %[[TILE_2_SLICE]], %[[DEST]][%[[TILE_2_I]], %[[C0]]] {in_bounds = [true]} : vector<[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT:   %[[TILE_3_I:.*]] = arith.addi %[[C12_VSCALE]], %[[I]] : index
  // CHECK-NEXT:   %[[TILE_3_SLICE:.*]] = vector.extract %[[TILE_3]][%[[I]]] : vector<[4]xf32> from vector<[4]x[4]xf32>
  // CHECK-NEXT:   vector.transfer_write %[[TILE_3_SLICE]], %[[DEST]][%[[TILE_3_I]], %[[C0]]] {in_bounds = [true]} : vector<[4]xf32>, memref<?x?xf32>
  // CHECK-NEXT: }
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

// -----

// CHECK-LABEL: @lift_illegal_transpose_to_memory(
// CHECK-SAME:                                    %[[INDEXA:[a-z0-9]+]]: index,
// CHECK-SAME:                                    %[[INDEXB:[a-z0-9]+]]: index,
// CHECK-SAME:                                    %[[MEMREF:[a-z0-9]+]]: memref<?x?xf32>)
func.func @lift_illegal_transpose_to_memory(%a: index, %b: index, %memref: memref<?x?xf32>) -> vector<4x[8]xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
  // CHECK-DAG: %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
  // CHECK-NEXT: %[[READ_SUBVIEW:.*]] = memref.subview %[[MEMREF]][%[[INDEXA]], %[[INDEXB]]] [%[[C8_VSCALE]], 4] [1, 1] : memref<?x?xf32> to memref<?x4xf32, strided<[?, 1], offset: ?>>
  // CHECK-NEXT: %[[CAST:.*]] = memref.cast %[[READ_SUBVIEW]] : memref<?x4xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = memref.transpose %[[CAST]] (d0, d1) -> (d1, d0) : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  // CHECK-NEXT: %[[LEGAL_READ:.*]]  = vector.transfer_read %[[TRANSPOSE]][%c0, %c0], %[[C0_F32]] : memref<?x?xf32, strided<[?, ?], offset: ?>>, vector<4x[8]xf32>
  // CHECK-NEXT: return %[[LEGAL_READ]]
  %pad = arith.constant 0.0 : f32
  %illegalRead = vector.transfer_read %memref[%a, %b], %pad : memref<?x?xf32>, vector<[8]x4xf32>
  %legalType = vector.transpose %illegalRead, [1, 0] : vector<[8]x4xf32> to vector<4x[8]xf32>
  return %legalType : vector<4x[8]xf32>
}

// -----

// CHECK-LABEL: @lift_illegal_transpose_to_memory_with_mask(
// CHECK-SAME:                                              %[[DIM0:[a-z0-9]+]]: index,
// CHECK-SAME:                                              %[[DIM1:[a-z0-9]+]]: index,
// CHECK-SAME:                                              %[[MEMREF:[a-z0-9]+]]: memref<?x?xf32>
func.func @lift_illegal_transpose_to_memory_with_mask(%dim0: index, %dim1: index, %memref: memref<?x?xf32>, %a: index, %b: index) -> vector<4x[8]xf32> {
  // CHECK-DAG: %[[READ_SUBVIEW:.*]] = memref.subview %[[MEMREF]]
  // CHECK-DAG: %[[CAST:.*]] = memref.cast %[[READ_SUBVIEW]]
  // CHECK-DAG: %[[TRANSPOSE:.*]] = memref.transpose %[[CAST]]
  // CHECK-DAG: %[[MASK:.*]] = vector.create_mask %[[DIM1]], %[[DIM0]] : vector<4x[8]xi1>
  // CHECK:     %[[LEGAL_READ:.*]] = vector.transfer_read %[[TRANSPOSE]]
  // CHECK-SAME:                       %[[MASK]] : memref<?x?xf32, strided<[?, ?], offset: ?>>, vector<4x[8]xf32>
  // CHECK-NEXT: return %[[LEGAL_READ]]
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %dim0, %dim1 : vector<[8]x4xi1>
  %illegalRead = vector.transfer_read %memref[%a, %b], %pad, %mask : memref<?x?xf32>, vector<[8]x4xf32>
  %legalType = vector.transpose %illegalRead, [1, 0] : vector<[8]x4xf32> to vector<4x[8]xf32>
  return %legalType : vector<4x[8]xf32>
}

// -----

// CHECK-LABEL: @lift_illegal_transpose_to_memory_with_arith_extop(
// CHECK-SAME:                                                     %[[MEMREF:[a-z0-9]+]]: memref<?x?xi8>
func.func @lift_illegal_transpose_to_memory_with_arith_extop(%a: index, %b: index, %memref: memref<?x?xi8>) -> vector<4x[8]xi32> {
  // CHECK-DAG: %[[READ_SUBVIEW:.*]] = memref.subview %[[MEMREF]]
  // CHECK-DAG: %[[CAST:.*]] = memref.cast %[[READ_SUBVIEW]]
  // CHECK-DAG: %[[TRANSPOSE:.*]] = memref.transpose %[[CAST]]
  // CHECK:     %[[LEGAL_READ:.*]] = vector.transfer_read %[[TRANSPOSE]]
  // CHECK-NEXT: %[[EXT_TYPE:.*]] = arith.extsi %[[LEGAL_READ]] : vector<4x[8]xi8> to vector<4x[8]xi32>
  // CHECK-NEXT: return %[[EXT_TYPE]]
  %pad = arith.constant 0 : i8
  %illegalRead = vector.transfer_read %memref[%a, %b], %pad : memref<?x?xi8>, vector<[8]x4xi8>
  %extRead = arith.extsi %illegalRead : vector<[8]x4xi8> to vector<[8]x4xi32>
  %legalType = vector.transpose %extRead, [1, 0] : vector<[8]x4xi32> to vector<4x[8]xi32>
  return %legalType : vector<4x[8]xi32>
}

// -----

// CHECK-LABEL: @lift_illegal_transpose_to_memory_with_in_bounds_attr
func.func @lift_illegal_transpose_to_memory_with_in_bounds_attr(%a: index, %b: index, %memref: memref<?x?xf32>) -> vector<4x[8]xf32> {
  // CHECK: vector.transfer_read
  // CHECK-SAME: in_bounds = [true, false]
  // CHECK-NOT: in_bounds = [false, true]
  %pad = arith.constant 0.0 : f32
  %illegalRead = vector.transfer_read %memref[%a, %b], %pad {in_bounds = [false, true]}: memref<?x?xf32>, vector<[8]x4xf32>
  %legalType = vector.transpose %illegalRead, [1, 0] : vector<[8]x4xf32> to vector<4x[8]xf32>
  return %legalType : vector<4x[8]xf32>
}

// -----

// The pass should do nothing (and not crash).
// CHECK-LABEL: @illegal_transpose_no_defining_source_op
func.func @illegal_transpose_no_defining_source_op(%vec: vector<[4]x1xf32>) -> vector<1x[4]xf32>
{
  // CHECK: vector.transpose
  %0 = vector.transpose %vec, [1, 0] : vector<[4]x1xf32> to vector<1x[4]xf32>
  return %0 : vector<1x[4]xf32>
}

// -----

// CHECK-LABEL: @illegal_shape_cast_to_transpose_2d(
// CHECK-SAME:                                      %[[VEC:.*]]: vector<[4]x1xf32>)
func.func @illegal_shape_cast_to_transpose_2d(%vec: vector<[4]x1xf32>) -> vector<1x[4]xf32> {
  // CHECK: vector.transpose %[[VEC]], [1, 0] : vector<[4]x1xf32> to vector<1x[4]xf32>
  %0 = vector.shape_cast %vec : vector<[4]x1xf32> to vector<1x[4]xf32>
  return %0 : vector<1x[4]xf32>
}

// -----

// CHECK-LABEL: @illegal_shape_cast_to_transpose_1d(
// CHECK-SAME:                                      %[[VEC:.*]]: vector<[4]x1xf32>)
func.func @illegal_shape_cast_to_transpose_1d(%vec: vector<[4]x1xf32>) -> vector<[4]xf32> {
  // CHECK: %[[TRANSPOSE:.*]] = vector.transpose %[[VEC]], [1, 0] : vector<[4]x1xf32> to vector<1x[4]xf32>
  // CHECK: vector.shape_cast %[[TRANSPOSE]] : vector<1x[4]xf32> to vector<[4]xf32>
  %0 = vector.shape_cast %vec : vector<[4]x1xf32> to vector<[4]xf32>
  return %0 : vector<[4]xf32>
}

// -----

// CHECK-LABEL: @lift_illegal_2d_shape_cast_to_memory
func.func @lift_illegal_2d_shape_cast_to_memory(%a: index, %b: index, %memref: memref<?x?xf32>) -> vector<1x[4]xf32> {
  // CHECK: vector.transfer_read {{.*}} : memref<?x?xf32, {{.*}}>, vector<1x[4]xf32>
  // CHECK-NOT: vector.shape_cast
  %pad = arith.constant 0.0 : f32
  %illegalRead = vector.transfer_read %memref[%a, %b], %pad {in_bounds = [false, true]}: memref<?x?xf32>, vector<[4]x1xf32>
  %cast = vector.shape_cast %illegalRead : vector<[4]x1xf32> to vector<1x[4]xf32>
  return %cast : vector<1x[4]xf32>
}

// -----

// CHECK-LABEL: @lift_illegal_1d_shape_cast_to_memory
func.func @lift_illegal_1d_shape_cast_to_memory(%a: index, %b: index, %memref: memref<?x?xf32>) -> vector<[4]xf32> {
  // CHECK: vector.transfer_read {{.*}} : memref<?x?xf32, {{.*}}>, vector<1x[4]xf32>
  // CHECK-NOT: vector.shape_cast {{.*}} : vector<[4]x1xf32> to vector<[4]xf32>
  %pad = arith.constant 0.0 : f32
  %illegalRead = vector.transfer_read %memref[%a, %b], %pad {in_bounds = [false, true]}: memref<?x?xf32>, vector<[4]x1xf32>
  %cast = vector.shape_cast %illegalRead : vector<[4]x1xf32> to vector<[4]xf32>
  return %cast : vector<[4]xf32>
}

// -----

// CHECK-LABEL: @multi_tile_splat
func.func @multi_tile_splat() -> vector<[8]x[8]xi32>
{
  // CHECK: %[[SPLAT:.*]] = arith.constant dense<42> : vector<[4]x[4]xi32>
  // CHECK-NEXT: return %[[SPLAT]], %[[SPLAT]], %[[SPLAT]], %[[SPLAT]] : vector<[4]x[4]xi32>, vector<[4]x[4]xi32>, vector<[4]x[4]xi32>, vector<[4]x[4]xi32>
  %0 = arith.constant dense<42> : vector<[8]x[8]xi32>
  return %0 : vector<[8]x[8]xi32>
}

// -----

// CHECK: #[[$TRANSPOSE_MAP_0:.*]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @transpose_store_scalable_via_za(
// CHECK-SAME:                                   %[[VEC:.*]]: vector<2x[4]xf32>
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xf32>,
// CHECK-SAME:                                   %[[I:.*]]: index,
// CHECK-SAME:                                   %[[J:.*]]: index)
func.func @transpose_store_scalable_via_za(%vec: vector<2x[4]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-NEXT: %[[INIT:.*]] = arm_sme.get_tile : vector<[4]x[4]xf32>
  // CHECK-NEXT: %[[V0:.*]] = vector.extract %[[VEC]][0] : vector<[4]xf32> from vector<2x[4]xf32>
  // CHECK-NEXT: %[[R0:.*]] = vector.insert %[[V0]], %[[INIT]] [0] : vector<[4]xf32> into vector<[4]x[4]xf32>
  // CHECK-NEXT: %[[V1:.*]] = vector.extract %[[VEC]][1] : vector<[4]xf32> from vector<2x[4]xf32>
  // CHECK-NEXT: %[[RES:.*]] = vector.insert %[[V1]], %[[R0]] [1] : vector<[4]xf32> into vector<[4]x[4]xf32>
  // CHECK-NEXT: %[[VSCALE:.*]] = vector.vscale
  // CHECK-NEXT: %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  // CHECK-NEXT: %[[MASK:.*]] = vector.create_mask %[[C4_VSCALE]], %[[C2]] : vector<[4]x[4]xi1>
  // CHECK-NEXT: vector.transfer_write %[[RES]], %[[DEST]][%[[I]], %[[J]]], %[[MASK]] {in_bounds = [true, true], permutation_map = #[[$TRANSPOSE_MAP_0]]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  %tr = vector.transpose %vec, [1, 0] : vector<2x[4]xf32> to vector<[4]x2xf32>
  vector.transfer_write %tr, %dest[%i, %j] {in_bounds = [true, true]} : vector<[4]x2xf32>,  memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_store_scalable_via_za_masked(
// CHECK-SAME:                                          %[[A:[a-z0-9]+]]: index,
// CHECK-SAME:                                          %[[B:[a-z0-9]+]]: index)
func.func @transpose_store_scalable_via_za_masked(%vec: vector<2x[4]xf32>, %dest: memref<?x?xf32>, %a: index, %b: index) {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[MIN:.*]] = index.mins %[[B]], %[[C2]]
  // CHECK: %[[MASK:.*]] = vector.create_mask %[[A]], %[[MIN]] : vector<[4]x[4]xi1>
  // CHECK: vector.transfer_write {{.*}} %[[MASK]] {{.*}} : vector<[4]x[4]xf32>, memref<?x?xf32>
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %a, %b : vector<[4]x2xi1>
  %tr = vector.transpose %vec, [1, 0] : vector<2x[4]xf32> to vector<[4]x2xf32>
  vector.transfer_write %tr, %dest[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[4]x2xf32>,  memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_store_scalable_via_za_multi_tile(
// CHECK-SAME:                                              %[[VEC:.*]]: vector<8x[4]xf32>
// CHECK-SAME:                                              %[[DEST:.*]]: memref<?x?xf32>,
// CHECK-SAME:                                              %[[I:.*]]: index,
// CHECK-SAME:                                              %[[J:.*]]: index)
func.func @transpose_store_scalable_via_za_multi_tile(%vec: vector<8x[4]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  // CHECK: %[[C4:.*]] = arith.constant 4 : index

  // <skip 3x other extract+insert chain>
  // CHECK: %[[V3:.*]] = vector.extract %[[VEC]][3] : vector<[4]xf32> from vector<8x[4]xf32>
  // CHECK: %[[TILE_0:.*]] = vector.insert %[[V3]], %{{.*}} [3] : vector<[4]xf32> into vector<[4]x[4]xf32>
  // CHECK: %[[VSCALE:.*]] = vector.vscale
  // CHECK: %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
  // CHECK: %[[MASK:.*]] = vector.create_mask %c4_vscale, %c4 : vector<[4]x[4]xi1>
  // CHECK: vector.transfer_write %[[TILE_0]], %[[DEST]][%[[I]], %[[J]]], %[[MASK]] {{.*}} : vector<[4]x[4]xf32>, memref<?x?xf32>

  // <skip 3x other extract+insert chain>
  // CHECK: %[[V7:.*]] = vector.extract %arg0[7] : vector<[4]xf32> from vector<8x[4]xf32>
  // CHECK: %[[TILE_1:.*]] = vector.insert %[[V7]], %{{.*}} [3] : vector<[4]xf32> into vector<[4]x[4]xf32>
  // CHECK: %[[J_OFFSET:.*]] = arith.addi %[[J]], %[[C4]] : index
  // CHECK: vector.transfer_write %[[TILE_1]], %[[DEST]][%[[I]], %[[J_OFFSET]]], %[[MASK]] {{.*}} : vector<[4]x[4]xf32>, memref<?x?xf32>
  %tr = vector.transpose %vec, [1, 0] : vector<8x[4]xf32> to vector<[4]x8xf32>
  vector.transfer_write %tr, %dest[%i, %j] {in_bounds = [true, true]} : vector<[4]x8xf32>,  memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_store_scalable_via_za_multi_tile_wide
func.func @transpose_store_scalable_via_za_multi_tile_wide(%vec: vector<2x[8]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  // <check extracts from lower 4 x vscale of %vec>
  // CHECK: vector.scalable.extract
  // CHECK: %[[ROW_2_LOWER:.*]] = vector.scalable.extract %{{.*}}[0] : vector<[4]xf32> from vector<[8]xf32>
  // CHECK: %[[TILE_0:.*]] = vector.insert %[[ROW_2_LOWER]], %{{.*}}[1] : vector<[4]xf32> into vector<[4]x[4]xf32>
  // CHECK: vector.transfer_write %[[TILE_0]], %{{.*}}[%[[I:.[a-z0-9]+]], %[[J:[a-z0-9]+]]]

  // <check extracts from upper 4 x vscale of %vec>
  // CHECK: vector.scalable.extract
  // CHECK: %[[ROW_2_UPPER:.*]] = vector.scalable.extract %{{.*}}[4] : vector<[4]xf32> from vector<[8]xf32>
  // CHECK: %[[TILE_0:.*]] = vector.insert %[[ROW_2_UPPER]], %{{.*}}[1] : vector<[4]xf32> into vector<[4]x[4]xf32>
  // CHECK: %[[I_OFFSET:.*]] = arith.addi %c4_vscale, %[[I]] : index
  // CHECK: vector.transfer_write %[[TILE_0]], %{{.*}}[%[[I_OFFSET]], %[[J]]]
  %tr = vector.transpose %vec, [1, 0] : vector<2x[8]xf32> to vector<[8]x2xf32>
  vector.transfer_write %tr, %dest[%i, %j] {in_bounds = [true, true]} : vector<[8]x2xf32>,  memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @negative_transpose_store_scalable_via_za__bad_source_shape
// CHECK-NOT: arm_sme.get_tile
func.func @negative_transpose_store_scalable_via_za__bad_source_shape(%vec: vector<2x[7]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  %tr = vector.transpose %vec, [1, 0] : vector<2x[7]xf32> to vector<[7]x2xf32>
  vector.transfer_write %tr, %dest[%i, %j] {in_bounds = [true, true]} : vector<[7]x2xf32>,  memref<?x?xf32>
  return
}
