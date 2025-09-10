// RUN: mlir-opt %s -convert-arm-sme-to-scf -cse -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// arm_sme.tile_load
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @arm_sme_tile_load_hor(
// CHECK-SAME:                                   %[[SRC:.*]]: memref<?x?xi32>) {
// CHECK-DAG:     %[[INIT_TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[VSCALE:.*]] = vector.vscale
// CHECK-DAG:     %[[PTRUE_S:.*]] = arith.constant dense<true> : vector<[4]xi1>
// CHECK-DAG:     %[[NUM_TILE_SLICES:.*]] = arith.muli %[[C4]], %[[VSCALE]] : index
// CHECK-NEXT:    scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[NUM_TILE_SLICES]] step %[[C1]] iter_args(%[[CURRENT_TILE:.*]] = %[[INIT_TILE]]) -> (vector<[4]x[4]xi32>) {
// CHECK-NEXT:      %[[OFFSET:.*]] = arith.addi %[[C0]], %[[TILE_SLICE_INDEX]] : index
// CHECK-NEXT:      %[[TILE_UPDATE:.*]] = arm_sme.load_tile_slice %[[SRC]]{{\[}}%[[OFFSET]], %[[C0]]], %[[PTRUE_S]], %[[CURRENT_TILE]], %[[TILE_SLICE_INDEX]] : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
// CHECK-NEXT:      scf.yield %[[TILE_UPDATE]] : vector<[4]x[4]xi32>
func.func @arm_sme_tile_load_hor(%src : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  "test.some_use" (%tile) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: @arm_sme_tile_load_ver
// CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical>
func.func @arm_sme_tile_load_ver(%src : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
  "test.some_use" (%tile) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @arm_sme_tile_load_hor_with_mask_and_pad_zero(
// CHECK-SAME:                                                          %[[SRC:.*]]: memref<?x?xi32>) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[NUM_ROWS:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[VSCALE:.*]] = vector.vscale
// CHECK-DAG:     %[[NUM_TILE_SLICES:.*]] = arith.muli %[[C4]], %[[VSCALE]] : index
// CHECK-DAG:     %[[NUM_ROWS_I64:.*]] = arith.index_cast %[[NUM_ROWS]] : index to i64
// CHECK-DAG:     %[[NUM_TILE_SLICES_I64:.*]] = arith.index_cast %[[NUM_TILE_SLICES]] : index to i64
// CHECK-DAG:     %[[LOOP_UPPER_BOUND_I64:.*]] = arith.minsi %[[NUM_ROWS_I64]], %[[NUM_TILE_SLICES_I64]] : i64
// CHECK-DAG:     %[[LOOP_UPPER_BOUND:.*]] = arith.index_cast %[[LOOP_UPPER_BOUND_I64]] : i64 to index
// CHECK-DAG:     %[[NUM_COLS:.*]] = vector.create_mask %c2 : vector<[4]xi1>
// CHECK-DAG:     %[[TILE_ZERO:.*]] = arm_sme.zero : vector<[4]x[4]xi32>
// CHECK-NEXT:    scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[LOOP_UPPER_BOUND]] step %[[C1]] iter_args(%[[CURRENT_TILE:.*]] = %[[TILE_ZERO]]) -> (vector<[4]x[4]xi32>) {
// CHECK-NEXT:      %[[OFFSET:.*]] = arith.addi %[[C0]], %[[TILE_SLICE_INDEX]] : index
// CHECK-NEXT:      %[[TILE_UPDATE:.*]] = arm_sme.load_tile_slice %[[SRC]]{{\[}}%[[OFFSET]], %[[C0]]], %[[NUM_COLS]], %[[CURRENT_TILE]], %[[TILE_SLICE_INDEX]] : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
// CHECK-NEXT:      scf.yield %[[TILE_UPDATE]] : vector<[4]x[4]xi32>
func.func @arm_sme_tile_load_hor_with_mask_and_pad_zero(%src : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %pad = arith.constant 0 : i32
  %mask = vector.create_mask %c3, %c2 : vector<[4]x[4]xi1>
  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask : memref<?x?xi32>, vector<[4]x[4]xi32>
  "test.some_use" (%tile) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @arm_sme_tile_load_hor_with_mask_and_nonzero_pad(
// CHECK-SAME:                                                             %[[SRC:.*]]: memref<?x?xi32>,
// CHECK-SAME:                                                             %[[PAD:.*]]: i32) {
// CHECK-DAG:     %[[TILE:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[NUM_ROWS:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[NUM_COLS:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[NUM_COLS_I32:.*]] = arith.index_castui %[[NUM_COLS]] : index to i32
// CHECK-DAG:     %[[VSCALE:.*]] = vector.vscale
// CHECK-NEXT:    %[[NUM_TILE_SLICES:.*]] = arith.muli %[[C4]], %[[VSCALE]] : index
// CHECK-NEXT:    scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[NUM_TILE_SLICES]] step %[[C1]] iter_args(%[[CURRENT_TILE:.*]] = %[[TILE]]) -> (vector<[4]x[4]xi32>) {
// CHECK-NEXT:        %[[ROW_IS_ACTIVE:.*]] = arith.cmpi slt, %[[TILE_SLICE_INDEX]], %[[NUM_ROWS]] : index
// CHECK-NEXT:        %[[ROW_IS_ACTIVE_SEXT_I32:.*]] = arith.extsi %[[ROW_IS_ACTIVE]] : i1 to i32
// CHECK-NEXT:        %[[MASK:.*]] = arith.andi %[[ROW_IS_ACTIVE_SEXT_I32]], %[[NUM_COLS_I32]] : i32
// CHECK-NEXT:        %[[MASK_INDEX:.*]] = arith.index_cast %[[MASK]] : i32 to index
// CHECK-NEXT:        %[[MASK_1D:.*]] = vector.create_mask %[[MASK_INDEX]] : vector<[4]xi1>
// CHECK-NEXT:        %[[OFFSET:.*]] = arith.addi %[[C0]], %[[TILE_SLICE_INDEX]] : index
// CHECK:             %[[PAD_1D:.*]] = vector.broadcast %[[PAD]] : i32 to vector<[4]xi32>
// CHECK:             %[[LOAD_SLICE:.*]] = vector.maskedload %[[SRC]]{{\[}}%[[OFFSET]], %[[C0]]], %[[MASK_1D]], %[[PAD_1D]] : memref<?x?xi32>, vector<[4]xi1>, vector<[4]xi32> into vector<[4]xi32>
// CHECK:             %[[TILE_UPDATE:.*]] = arm_sme.insert_tile_slice %[[LOAD_SLICE]], %[[CURRENT_TILE]][%[[TILE_SLICE_INDEX]]] : vector<[4]xi32> into vector<[4]x[4]xi32>
// CHECK-NEXT:        scf.yield %[[TILE_UPDATE]] : vector<[4]x[4]xi32>
func.func @arm_sme_tile_load_hor_with_mask_and_nonzero_pad(%src : memref<?x?xi32>, %pad : i32) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %mask = vector.create_mask %c3, %c2 : vector<[4]x[4]xi1>
  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask : memref<?x?xi32>, vector<[4]x[4]xi32>
  "test.some_use" (%tile) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

func.func @arm_sme_tile_load_zero_pad__unsupported_mask_op(%src : memref<?x?xi32>, %mask : vector<[4]x[4]xi1>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i32
  // expected-error@+1 {{failed to legalize operation 'arm_sme.tile_load' that was explicitly marked illegal}}
  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask : memref<?x?xi32>, vector<[4]x[4]xi32>
  "test.some_use" (%tile) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

func.func @arm_sme_tile_load_nonzero_pad__unsupported_mask_op(%src : memref<?x?xi32>, %pad : i32, %mask : vector<[4]x[4]xi1>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{failed to legalize operation 'arm_sme.tile_load' that was explicitly marked illegal}}
  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask : memref<?x?xi32>, vector<[4]x[4]xi32>
  "test.some_use" (%tile) : (vector<[4]x[4]xi32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.tile_store
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func.func @arm_sme_tile_store_hor(
// CHECK-SAME:                                    %[[TILE:.*]]: vector<[4]x[4]xi32>,
// CHECK-SAME:                                    %[[DEST:.*]]: memref<?x?xi32>) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[VSCALE:.*]] = vector.vscale
// CHECK-DAG:     %[[PTRUE_S:.*]] = arith.constant dense<true> : vector<[4]xi1>
// CHECK-DAG:     %[[NUM_TILE_SLICES:.*]] = arith.muli %[[C4]], %[[VSCALE]] : index
// CHECK:         scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[NUM_TILE_SLICES]] step %[[C1]] {
// CHECK:           %[[OFFSET:.*]] = arith.addi %[[C0]], %[[TILE_SLICE_INDEX]] : index
// CHECK:           arm_sme.store_tile_slice %[[TILE]], %[[TILE_SLICE_INDEX]], %[[PTRUE_S]], %[[DEST]]{{\[}}%[[OFFSET]], %[[C0]]] : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
func.func @arm_sme_tile_store_hor(%tile : vector<[4]x[4]xi32>, %dest : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @arm_sme_tile_store_ver
// CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical>
func.func @arm_sme_tile_store_ver(%tile : vector<[4]x[4]xi32>, %dest : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: func.func @arm_sme_tile_store_hor_with_mask(
// CHECK-SAME:                                             %[[TILE:.*]]: vector<[4]x[4]xi32>,
// CHECK-SAME:                                             %[[DEST:.*]]: memref<?x?xi32>) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[NUM_ROWS:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[VSCALE:.*]] = vector.vscale
// CHECK-DAG:     %[[NUM_TILE_SLICES:.*]] = arith.muli %[[C4]], %[[VSCALE]] : index
// CHECK-DAG:     %[[NUM_ROWS_I64:.*]] = arith.index_cast %[[NUM_ROWS]] : index to i64
// CHECK-DAG:     %[[NUM_TILE_SLICES_I64:.*]] = arith.index_cast %[[NUM_TILE_SLICES]] : index to i64
// CHECK-DAG:     %[[LOOP_UPPER_BOUND_I64:.*]] = arith.minsi %[[NUM_ROWS_I64]], %[[NUM_TILE_SLICES_I64]] : i64
// CHECK-DAG:     %[[LOOP_UPPER_BOUND:.*]] = arith.index_cast %[[LOOP_UPPER_BOUND_I64]] : i64 to index
// CHECK-DAG:     %[[NUM_COLS:.*]] = vector.create_mask %c2 : vector<[4]xi1>
// CHECK-NEXT:    scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[LOOP_UPPER_BOUND]] step %[[C1]] {
// CHECK-NEXT:      %[[OFFSET:.*]] = arith.addi %[[C0]], %[[TILE_SLICE_INDEX]] : index
// CHECK-NEXT:      arm_sme.store_tile_slice %[[TILE]], %[[TILE_SLICE_INDEX]], %[[NUM_COLS]], %[[DEST]]{{\[}}%[[OFFSET]], %[[C0]]] : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
func.func @arm_sme_tile_store_hor_with_mask(%tile : vector<[4]x[4]xi32>, %dest : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %mask = vector.create_mask %c3, %c2 : vector<[4]x[4]xi1>
  arm_sme.tile_store %tile, %dest[%c0, %c0], %mask : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}
