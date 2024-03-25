// RUN: mlir-opt %s -convert-vector-to-arm-sme -convert-arith-to-arm-sme -allocate-arm-sme-tiles -convert-arm-sme-to-scf -convert-arm-sme-to-llvm -cse -canonicalize -split-input-file -allow-unregistered-dialect -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// vector.transfer_write
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @transfer_write_2d_zero_i8(
// CHECK-SAME:                             %[[ARG0:.*]]: memref<?x?xi8>)
// CHECK-DAG:  %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[MIN_SVL_B:.*]] = arith.constant 16 : index
// CHECK-DAG:  %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[16]xi1>
// CHECK-DAG:  %[[C0_I64:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to i64
// CHECK-DAG:  "arm_sme.intr.zero"() <{tile_mask = 255 : i32}> : () -> ()
// CHECK-DAG:  %[[VSCALE:.*]] = vector.vscale
// CHECK-NEXT: %[[SVL_B:.*]] = arith.muli %[[VSCALE]], %[[MIN_SVL_B]] : index
// CHECK-NEXT: scf.for %[[TILE_SLICE:.*]] = %[[C0]] to %[[SVL_B]] step %[[C1]] {
// CHECK:        %[[TILE_SLICE_I64:.*]] = builtin.unrealized_conversion_cast %[[TILE_SLICE]] : index to i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[OFF0:.*]] = llvm.mul %[[TILE_SLICE_I64]], %[[STRIDE0]]  : i64
// CHECK-NEXT:   %[[OFF1:.*]] = llvm.add %[[OFF0]], %[[C0_I64]]  : i64
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFF1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[TILE_SLICE_I32:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i32
// CHECK-NEXT:   "arm_sme.intr.st1b.horiz"(%[[PTRUE_ALL]], %[[GEP]], %[[TILE_SLICE_I32]]) <{tile_id = 0 : i32}> : (vector<[16]xi1>, !llvm.ptr, i32) -> ()
func.func @transfer_write_2d_zero_i8(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

//===----------------------------------------------------------------------===//
// vector.load
//===----------------------------------------------------------------------===//

// -----

// Load an 8-bit tile from a rank 2 memref with a non-zero offset for the first
// memref index. This verifies the offset is preserved when materializing the
// loop of tile slice loads.

// CHECK-LABEL: @vector_load_i8_with_offset(
// CHECK-SAME:                              %[[ARG0:.*]]: memref<?x?xi8>)
// CHECK-DAG:  %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[C123:.*]] = arith.constant 123 : index
// CHECK-DAG:  %[[MIN_SVL_B:.*]] = arith.constant 16 : index
// CHECK-DAG:  %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[16]xi1>
// CHECK-DAG:  %[[C0_I64:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to i64
// CHECK-DAG:  %[[VSCALE:.*]] = vector.vscale
// CHECK-NEXT: %[[SVL_B:.*]] = arith.muli %[[VSCALE]], %[[MIN_SVL_B]] : index
// CHECK-NEXT: scf.for %[[TILE_SLICE:.*]] = %[[C0]] to %[[SVL_B]] step %[[C1]] {
// CHECK-NEXT:   %[[TILE_SLICE_PLUS_OFF0:.*]] = arith.addi %[[TILE_SLICE]], %[[C123]] : index
// CHECK-NEXT:   %[[TILE_SLICE_PLUS_OFF0_I64:.*]] = builtin.unrealized_conversion_cast %[[TILE_SLICE_PLUS_OFF0]] : index to i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[OFF0:.*]] = llvm.mul %[[TILE_SLICE_PLUS_OFF0_I64]], %[[STRIDE0]]  : i64
// CHECK-NEXT:   %[[OFF1:.*]] = llvm.add %[[OFF0]], %[[C0_I64]]  : i64
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFF1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[TILE_SLICE_I32:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i32
// CHECK-NEXT:   "arm_sme.intr.ld1b.horiz"(%[[PTRUE_ALL]], %[[GEP]], %[[TILE_SLICE_I32]]) <{tile_id = 0 : i32}> : (vector<[16]xi1>, !llvm.ptr, i32) -> ()
// CHECK-NEXT: }
func.func @vector_load_i8_with_offset(%arg0 : memref<?x?xi8>) -> vector<[16]x[16]xi8> {
  %c0 = arith.constant 0 : index
  %c123 = arith.constant 123 : index
  %tile = vector.load %arg0[%c123, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return %tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @vector_load_i8_from_rank_1_memref(
// CHECK-SAME:                                     %[[ARG0:.*]]: memref<?xi8>)
// CHECK-DAG:  %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?xi8> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[MIN_SVL_B:.*]] = arith.constant 16 : index
// CHECK-DAG:  %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[16]xi1>
// CHECK-DAG:  %[[VSCALE:.*]] = vector.vscale
// CHECK-NEXT: %[[SVL_B:.*]] = arith.muli %[[VSCALE]], %[[MIN_SVL_B]] : index
// CHECK-NEXT: scf.for %[[TILE_SLICE:.*]] = %[[C0]] to %[[SVL_B]] step %[[C1]] {
// CHECK-NEXT:   %[[TILE_SLICE_IDX:.*]] = arith.muli %[[TILE_SLICE]], %[[SVL_B]] : index
// CHECK-NEXT:   %[[TILE_SLICE_IDX_I64:.*]] = builtin.unrealized_conversion_cast %[[TILE_SLICE_IDX]] : index to i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[TILE_SLICE_IDX_I64]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[TILE_SLICE_I32:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i32
// CHECK-NEXT:   "arm_sme.intr.ld1b.horiz"(%[[PTRUE_ALL]], %[[GEP]], %[[TILE_SLICE_I32]]) <{tile_id = 0 : i32}> : (vector<[16]xi1>, !llvm.ptr, i32) -> ()
// CHECK-NEXT: }
func.func @vector_load_i8_from_rank_1_memref(%arg0 : memref<?xi8>) -> vector<[16]x[16]xi8> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0] : memref<?xi8>, vector<[16]x[16]xi8>
  return %tile : vector<[16]x[16]xi8>
}


// -----

// CHECK-LABEL: @vector_load_i16(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi16>)
// CHECK-DAG: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK:     %[[SVL_H:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_H]] : index
// CHECK:       arm_sme.intr.ld1h.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_load_i16(%arg0 : memref<?x?xi16>) -> vector<[8]x[8]xi16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return %tile : vector<[8]x[8]xi16>
}

// -----

// CHECK-LABEL: @vector_load_i32(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi32>)
// CHECK-DAG: %[[MIN_SVL_S:.*]] = arith.constant 4 : index
// CHECK:     %[[SVL_S:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_S]] : index
// CHECK:       arm_sme.intr.ld1w.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_load_i32(%arg0 : memref<?x?xi32>) -> vector<[4]x[4]xi32> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return %tile : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @vector_load_i64(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi64>)
// CHECK-DAG: %[[MIN_SVL_D:.*]] = arith.constant 2 : index
// CHECK:     %[[SVL_D:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_D]] : index
// CHECK:       arm_sme.intr.ld1d.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_load_i64(%arg0 : memref<?x?xi64>) -> vector<[2]x[2]xi64> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return %tile : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @vector_load_f16(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf16>)
// CHECK-DAG: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK:     %[[SVL_H:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_H]] : index
// CHECK:       arm_sme.intr.ld1h.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_load_f16(%arg0 : memref<?x?xf16>) -> vector<[8]x[8]xf16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return %tile : vector<[8]x[8]xf16>
}

// -----

// CHECK-LABEL: @vector_load_bf16(
// CHECK-SAME:                    %[[ARG0:.*]]: memref<?x?xbf16>)
// CHECK-DAG: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK:     %[[SVL_H:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_H]] : index
// CHECK:       arm_sme.intr.ld1h.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_load_bf16(%arg0 : memref<?x?xbf16>) -> vector<[8]x[8]xbf16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return %tile : vector<[8]x[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_load_f32(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf32>)
// CHECK-DAG: %[[MIN_SVL_S:.*]] = arith.constant 4 : index
// CHECK:     %[[SVL_S:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_S]] : index
// CHECK:       arm_sme.intr.ld1w.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_load_f32(%arg0 : memref<?x?xf32>) -> vector<[4]x[4]xf32> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return %tile : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @vector_load_f64(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf64>)
// CHECK-DAG: %[[MIN_SVL_D:.*]] = arith.constant 2 : index
// CHECK:     %[[SVL_D:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_D]] : index
// CHECK:       arm_sme.intr.ld1d.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_load_f64(%arg0 : memref<?x?xf64>) -> vector<[2]x[2]xf64> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return %tile : vector<[2]x[2]xf64>
}

// -----

// CHECK-LABEL: @vector_load_i128(
// CHECK-SAME:                    %[[ARG0:.*]]: memref<?x?xi128>)
// CHECK:       arm_sme.intr.ld1q.horiz
// CHECK-SAME:  tile_id = 0
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
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi8>)
// CHECK-DAG:  %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[MIN_SVL_B:.*]] = arith.constant 16 : index
// CHECK-DAG:  %[[C0_I64:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to i64
// CHECK-DAG:  %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[16]xi1>
// CHECK-DAG:  %[[VSCALE:.*]] = vector.vscale
// CHECK-NEXT: %[[SVL_B:.*]] = arith.muli %[[VSCALE]], %[[MIN_SVL_B]] : index
// CHECK-NEXT: scf.for %[[TILE_SLICE:.*]] = %[[C0]] to %[[SVL_B]] step %[[C1]] {
// CHECK:        %[[TILE_SLICE_I64:.*]] = builtin.unrealized_conversion_cast %[[TILE_SLICE]] : index to i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[OFF0:.*]] = llvm.mul %[[TILE_SLICE_I64]], %[[STRIDE0]]  : i64
// CHECK-NEXT:   %[[OFF1:.*]] = llvm.add %[[OFF0]], %[[C0_I64]]  : i64
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFF1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[TILE_SLICE_I32:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i32
// CHECK-NEXT:   "arm_sme.intr.st1b.horiz"(%[[PTRUE_ALL]], %[[GEP]], %[[TILE_SLICE_I32]]) <{tile_id = 0 : i32}> : (vector<[16]xi1>, !llvm.ptr, i32) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @vector_store_i8(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: @vector_store_i16(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi16>)
// CHECK:     %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK:     %[[SVL_H:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_H]] : index
// CHECK:       arm_sme.intr.st1h.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_store_i16(%arg0 : memref<?x?xi16>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return
}

// -----

// CHECK-LABEL: @vector_store_i32(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi32>)
// CHECK:     %[[MIN_SVL_S:.*]] = arith.constant 4 : index
// CHECK:     %[[SVL_S:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_S]] : index
// CHECK:       arm_sme.intr.st1w.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_store_i32(%arg0 : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @vector_store_i64(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi64>)
// CHECK:     %[[MIN_SVL_D:.*]] = arith.constant 2 : index
// CHECK:     %[[SVL_D:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_D]] : index
// CHECK:       arm_sme.intr.st1d.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_store_i64(%arg0 : memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return
}

// -----

// CHECK-LABEL: @vector_store_f16(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf16>)
// CHECK:     %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK:     %[[SVL_H:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_H]] : index
// CHECK:       arm_sme.intr.st1h.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_store_f16(%arg0 : memref<?x?xf16>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return
}

// -----

// CHECK-LABEL: @vector_store_bf16(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xbf16>)
// CHECK:     %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK:     %[[SVL_H:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_H]] : index
// CHECK:       arm_sme.intr.st1h.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_store_bf16(%arg0 : memref<?x?xbf16>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return
}
// -----

// CHECK-LABEL: @vector_store_f32(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf32>)
// CHECK:     %[[MIN_SVL_S:.*]] = arith.constant 4 : index
// CHECK:     %[[SVL_S:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_S]] : index
// CHECK:       arm_sme.intr.st1w.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_store_f32(%arg0 : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

// CHECK-LABEL: @vector_store_f64(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf64>)
// CHECK:     %[[MIN_SVL_D:.*]] = arith.constant 2 : index
// CHECK:     %[[SVL_D:.*]] = arith.muli %{{.*}}, %[[MIN_SVL_D]] : index
// CHECK:       arm_sme.intr.st1d.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_store_f64(%arg0 : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

// CHECK-LABEL: @vector_store_i128(
// CHECK-SAME:                     %[[ARG0:.*]]: memref<?x?xi128>)
// CHECK:       arm_sme.intr.st1q.horiz
// CHECK-SAME:  tile_id = 0
func.func @vector_store_i128(%arg0 : memref<?x?xi128>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
  return
}

//===----------------------------------------------------------------------===//
// vector.outerproduct
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_outerproduct_add_f16
// CHECK-SAME: (%[[LHS:.*]]: vector<[8]xf16>, %[[RHS:.*]]: vector<[8]xf16>)
func.func @vector_outerproduct_add_f16(%lhs : vector<[8]xf16>, %rhs : vector<[8]xf16>) {
  // CHECK: %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[8]xi1>
  // CHECK: "arm_sme.intr.mopa"(%[[PTRUE_ALL]], %[[PTRUE_ALL]], %[[LHS]], %[[RHS]]) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>)
  %acc = arm_sme.get_tile : vector<[8]x[8]xf16>
  %0 = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[8]xf16>, vector<[8]xf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xf16>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_add_bf16
func.func @vector_outerproduct_add_bf16(%lhs : vector<[8]xbf16>, %rhs : vector<[8]xbf16>) {
  // CHECK: "arm_sme.intr.mopa"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>)
  %acc = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %0 = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[8]xbf16>, vector<[8]xbf16>
  "prevent.dce"(%0) : (vector<[8]x[8]xbf16>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_add_f32
func.func @vector_outerproduct_add_f32(%lhs : vector<[4]xf32>, %rhs : vector<[4]xf32>) {
  // CHECK: "arm_sme.intr.mopa"({{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>)
  %acc = arm_sme.get_tile : vector<[4]x[4]xf32>
  %0 = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32>
  "prevent.dce"(%0) : (vector<[4]x[4]xf32>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_add_f64
func.func @vector_outerproduct_add_f64(%lhs : vector<[2]xf64>, %rhs : vector<[2]xf64>) {
  // CHECK: "arm_sme.intr.mopa"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>)
  %acc = arm_sme.get_tile : vector<[2]x[2]xf64>
  %0 = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[2]xf64>, vector<[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_no_accumulator
func.func @vector_outerproduct_no_accumulator(%lhs : vector<[2]xf64>, %rhs : vector<[2]xf64>) {
  // CHECK: "arm_sme.intr.zero"() <{tile_mask = 1 : i32}> : () -> ()
  // CHECK: "arm_sme.intr.mopa"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>)
  %0 = vector.outerproduct %lhs, %rhs {kind = #vector.kind<add>} : vector<[2]xf64>, vector<[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_masked_f32
// CHECK-SAME: (%[[LHS:.*]]: vector<[4]xf32>, %[[RHS:.*]]: vector<[4]xf32>, %[[DIM0:.*]]: index, %[[DIM1:.*]]: index
func.func @vector_outerproduct_masked_f32(%lhs : vector<[4]xf32>, %rhs : vector<[4]xf32>, %dim0 : index, %dim1 : index) {
  // CHECK: %[[LHS_MASK:.*]] = vector.create_mask %[[DIM0]] : vector<[4]xi1>
  // CHECK: %[[RHS_MASK:.*]] = vector.create_mask %[[DIM1]] : vector<[4]xi1>
  // CHECK: "arm_sme.intr.mopa"(%[[LHS_MASK]], %[[RHS_MASK]], %[[LHS]], %[[RHS]]) <{tile_id = 0 : i32}> : (vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>)
  %acc = arm_sme.get_tile : vector<[4]x[4]xf32>
  %mask = vector.create_mask %dim0, %dim1 : vector<[4]x[4]xi1>
  %result = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32> } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>
  "prevent.dce"(%result) : (vector<[4]x[4]xf32>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_masked_f16
// CHECK-SAME: (%[[LHS:.*]]: vector<[8]xf16>, %[[RHS:.*]]: vector<[8]xf16>,
func.func @vector_outerproduct_masked_f16(%lhs : vector<[8]xf16>, %rhs : vector<[8]xf16>, %dim0 : index, %dim1 : index) {
  // CHECK: vector.create_mask {{.*}} : vector<[8]xi1>
  // CHECK: vector.create_mask {{.*}} : vector<[8]xi1>
  // CHECK: "arm_sme.intr.mopa"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>)
  %acc = arm_sme.get_tile : vector<[8]x[8]xf16>
  %mask = vector.create_mask %dim0, %dim1 : vector<[8]x[8]xi1>
  %result = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[8]xf16>, vector<[8]xf16> } : vector<[8]x[8]xi1> -> vector<[8]x[8]xf16>
  "prevent.dce"(%result) : (vector<[8]x[8]xf16>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_masked_bf16
// CHECK-SAME: (%[[LHS:.*]]: vector<[8]xbf16>, %[[RHS:.*]]: vector<[8]xbf16>
func.func @vector_outerproduct_masked_bf16(%lhs : vector<[8]xbf16>, %rhs : vector<[8]xbf16>, %dim0 : index, %dim1 : index) {
  // CHECK: vector.create_mask {{.*}} : vector<[8]xi1>
  // CHECK: vector.create_mask {{.*}} : vector<[8]xi1>
  // CHECK: "arm_sme.intr.mopa"({{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>)
  %acc = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %mask = vector.create_mask %dim0, %dim1 : vector<[8]x[8]xi1>
  %result = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[8]xbf16>, vector<[8]xbf16> } : vector<[8]x[8]xi1> -> vector<[8]x[8]xbf16>
  "prevent.dce"(%result) : (vector<[8]x[8]xbf16>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_masked_f64
// CHECK-SAME: (%[[LHS:.*]]: vector<[2]xf64>, %[[RHS:.*]]: vector<[2]xf64>,
func.func @vector_outerproduct_masked_f64(%lhs : vector<[2]xf64>, %rhs : vector<[2]xf64>, %dim0 : index, %dim1 : index) {
  // CHECK: vector.create_mask {{.*}} : vector<[2]xi1>
  // CHECK: vector.create_mask {{.*}} : vector<[2]xi1>
  // CHECK: "arm_sme.intr.mopa"({{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>)
  %acc = arm_sme.get_tile : vector<[2]x[2]xf64>
  %mask = vector.create_mask %dim0, %dim1 : vector<[2]x[2]xi1>
  %result = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[2]xf64>, vector<[2]xf64> } : vector<[2]x[2]xi1> -> vector<[2]x[2]xf64>
  "prevent.dce"(%result) : (vector<[2]x[2]xf64>) -> ()
}

//===----------------------------------------------------------------------===//
// vector.insert
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_insert_slice_i32(
// CHECK-SAME:                       %[[SLICE:.*]]: vector<[4]xi32>,
// CHECK-SAME:                       %[[INDEX:.*]]: index)
func.func @vector_insert_slice_i32(%slice: vector<[4]xi32>, %row: index) -> vector<[4]x[4]xi32>{
  // CHECK-NEXT: %[[PTRUE:.*]] = arith.constant dense<true> : vector<[4]xi1>
  // CHECK: %[[TILE_SLICE_INDEX:.*]] = arith.index_castui %[[INDEX]] : index to i32
  // CHECK-NEXT: "arm_sme.intr.write.horiz"(%[[TILE_SLICE_INDEX]], %[[PTRUE]], %[[SLICE]]) <{tile_id = 0 : i32}> : (i32, vector<[4]xi1>, vector<[4]xi32>) -> ()
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[4]xi32> into vector<[4]x[4]xi32>
  return %new_tile : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @vector_insert_slice_i8
func.func @vector_insert_slice_i8(%slice: vector<[16]xi8>, %row: index) -> vector<[16]x[16]xi8> {
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[16]xi1>, vector<[16]xi8>) -> ()
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[16]xi8> into vector<[16]x[16]xi8>
  return %new_tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @vector_insert_slice_i16
func.func @vector_insert_slice_i16(%slice: vector<[8]xi16>, %row: index) -> vector<[8]x[8]xi16> {
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[8]xi1>, vector<[8]xi16>) -> ()
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[8]xi16> into vector<[8]x[8]xi16>
  return %new_tile : vector<[8]x[8]xi16>
}

// -----

// CHECK-LABEL: @vector_insert_slice_i64
func.func @vector_insert_slice_i64(%slice: vector<[2]xi64>, %row: index) -> vector<[2]x[2]xi64> {
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[2]xi1>, vector<[2]xi64>) -> ()
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[2]xi64> into vector<[2]x[2]xi64>
  return %new_tile : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @vector_insert_slice_i128
func.func @vector_insert_slice_i128(%slice: vector<[1]xi128>, %row: index) -> vector<[1]x[1]xi128> {
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[1]xi1>, vector<[1]xi128>) -> ()
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[1]xi128> into vector<[1]x[1]xi128>
  return %new_tile : vector<[1]x[1]xi128>
}

// -----

// CHECK-LABEL: @vector_insert_slice_f16
func.func @vector_insert_slice_f16(%slice: vector<[8]xf16>, %row: index) -> vector<[8]x[8]xf16> {
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[8]xi1>, vector<[8]xf16>) -> ()
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[8]xf16> into vector<[8]x[8]xf16>
  return %new_tile : vector<[8]x[8]xf16>
}

// -----

// CHECK-LABEL: @vector_insert_slice_bf16
func.func @vector_insert_slice_bf16(%slice: vector<[8]xbf16>, %row: index) -> vector<[8]x[8]xbf16> {
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[8]xi1>, vector<[8]xbf16>) -> ()
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[8]xbf16> into vector<[8]x[8]xbf16>
  return %new_tile : vector<[8]x[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_insert_slice_f32
func.func @vector_insert_slice_f32(%slice: vector<[4]xf32>, %row: index) -> vector<[4]x[4]xf32> {
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[4]xi1>, vector<[4]xf32>) -> ()
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %new_tile = vector.insert %slice, %tile[%row] : vector<[4]xf32> into vector<[4]x[4]xf32>
  return %new_tile : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @vector_insert_slice_f64
func.func @vector_insert_slice_f64(%slice: vector<[2]xf64>, %row: index) -> vector<[2]x[2]xf64> {
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[2]xi1>, vector<[2]xf64>) -> ()
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
  // CHECK-DAG:  %[[ZERO_VEC:.*]] = arith.constant dense<0> : vector<[4]xi32>
  // CHECK-DAG:  %[[PTRUE:.*]] = arith.constant dense<true> : vector<[4]xi1>
  // CHECK-DAG:  %[[ROW_I32:.*]] = arith.index_cast %[[ROW]] : index to i32
  // CHECK-NEXT: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%[[ZERO_VEC]], %[[PTRUE]], %[[ROW_I32]]) <{tile_id = 0 : i32}> : (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>
  // CHECK-NEXT: %[[NEW_SLICE:.*]] = vector.insert %[[EL]], %[[SLICE]] [%[[COL]]] : i32 into vector<[4]xi32>
  // CHECK-NEXT: %[[SLICE_INDEX:.*]] = arith.index_castui %[[ROW]] : index to i32
  // CHECK-NEXT: "arm_sme.intr.write.horiz"(%[[SLICE_INDEX]], %[[PTRUE]], %[[NEW_SLICE]]) <{tile_id = 0 : i32}> : (i32, vector<[4]xi1>, vector<[4]xi32>) -> ()
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %new_tile = vector.insert %el, %tile[%row, %col] : i32 into vector<[4]x[4]xi32>
  return %new_tile : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @vector_insert_element_i8
func.func @vector_insert_element_i8(%el: i8, %row: index, %col: index) -> vector<[16]x[16]xi8> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[16]xi8>, vector<[16]xi1>, i32) -> vector<[16]xi8>
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[16]xi1>, vector<[16]xi8>) -> ()
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %new_tile = vector.insert %el, %tile[%row, %col] : i8 into vector<[16]x[16]xi8>
  return %new_tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @vector_insert_element_i16
func.func @vector_insert_element_i16(%el: i16, %row: index, %col: index) -> vector<[8]x[8]xi16> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi16>, vector<[8]xi1>, i32) -> vector<[8]xi16>
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[8]xi1>, vector<[8]xi16>) -> ()
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %new_tile = vector.insert %el, %tile[%row, %col] : i16 into vector<[8]x[8]xi16>
  return %new_tile : vector<[8]x[8]xi16>
}

// -----

// CHECK-LABEL: @vector_insert_element_i64
func.func @vector_insert_element_i64(%el: i64, %row: index, %col: index) -> vector<[2]x[2]xi64> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi64>, vector<[2]xi1>, i32) -> vector<[2]xi64>
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[2]xi1>, vector<[2]xi64>) -> ()
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %new_tile = vector.insert %el, %tile[%row, %col] : i64 into vector<[2]x[2]xi64>
  return %new_tile : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @vector_insert_element_i128
func.func @vector_insert_element_i128(%el: i128, %row: index, %col: index) -> vector<[1]x[1]xi128> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi128>, vector<[1]xi1>, i32) -> vector<[1]xi128>
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[1]xi1>, vector<[1]xi128>) -> ()
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %new_tile = vector.insert %el, %tile[%row, %col] : i128 into vector<[1]x[1]xi128>
  return %new_tile : vector<[1]x[1]xi128>
}

// -----

// CHECK-LABEL: @vector_insert_element_f16
func.func @vector_insert_element_f16(%el: f16, %row: index, %col: index) -> vector<[8]x[8]xf16> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xf16>, vector<[8]xi1>, i32) -> vector<[8]xf16>
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[8]xi1>, vector<[8]xf16>) -> ()
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %new_tile = vector.insert %el, %tile[%row, %col] : f16 into vector<[8]x[8]xf16>
  return %new_tile : vector<[8]x[8]xf16>
}

// -----

// CHECK-LABEL: @vector_insert_element_bf16
func.func @vector_insert_element_bf16(%el: bf16, %row: index, %col: index) -> vector<[8]x[8]xbf16> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xbf16>, vector<[8]xi1>, i32) -> vector<[8]xbf16>
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[8]xi1>, vector<[8]xbf16>) -> ()
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %new_tile = vector.insert %el, %tile[%row, %col] : bf16 into vector<[8]x[8]xbf16>
  return %new_tile : vector<[8]x[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_insert_element_f32
func.func @vector_insert_element_f32(%el: f32, %row: index, %col: index) -> vector<[4]x[4]xf32> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[4]xi1>, vector<[4]xf32>) -> ()
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %new_tile = vector.insert %el, %tile[%row, %col] : f32 into vector<[4]x[4]xf32>
  return %new_tile : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @vector_insert_element_f64
func.func @vector_insert_element_f64(%el: f64, %row: index, %col: index) -> vector<[2]x[2]xf64> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xf64>, vector<[2]xi1>, i32) -> vector<[2]xf64>
  // CHECK: "arm_sme.intr.write.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (i32, vector<[2]xi1>, vector<[2]xf64>) -> ()
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %new_tile = vector.insert %el, %tile[%row, %col] : f64 into vector<[2]x[2]xf64>
  return %new_tile : vector<[2]x[2]xf64>
}

//===----------------------------------------------------------------------===//
// vector.extract
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_extract_slice_i32(
// CHECK-SAME:                        %[[INDEX:.*]]: index)
func.func @vector_extract_slice_i32(%row: index) -> vector<[4]xi32> {
  // CHECK-NEXT: %[[PTRUE:.*]] = arith.constant dense<true> : vector<[4]xi1>
  // CHECK-NEXT: %[[ZERO_VEC:.*]] = arith.constant dense<0> : vector<[4]xi32>
  // CHECK-NEXT: %[[TILE_SLICE_INDEX:.*]] = arith.index_cast %[[INDEX]] : index to i32
  // CHECK-NEXT: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%[[ZERO_VEC]], %[[PTRUE]], %[[TILE_SLICE_INDEX]]) <{tile_id = 0 : i32}> : (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %slice = vector.extract %tile[%row] : vector<[4]xi32> from vector<[4]x[4]xi32>
  return %slice : vector<[4]xi32>
}

// -----

// CHECK-LABEL: @vector_extract_slice_i8
func.func @vector_extract_slice_i8(%row: index) -> vector<[16]xi8> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[16]xi8>, vector<[16]xi1>, i32) -> vector<[16]xi8>
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %slice = vector.extract %tile[%row] : vector<[16]xi8> from vector<[16]x[16]xi8>
  return %slice : vector<[16]xi8>
}

// -----

// CHECK-LABEL: @vector_extract_slice_i16
func.func @vector_extract_slice_i16(%row: index) -> vector<[8]xi16> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi16>, vector<[8]xi1>, i32) -> vector<[8]xi16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %slice = vector.extract %tile[%row] : vector<[8]xi16> from vector<[8]x[8]xi16>
  return %slice : vector<[8]xi16>
}

// -----

// CHECK-LABEL: @vector_extract_slice_i64
func.func @vector_extract_slice_i64(%row: index) -> vector<[2]xi64> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi64>, vector<[2]xi1>, i32) -> vector<[2]xi64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %slice = vector.extract %tile[%row] : vector<[2]xi64> from vector<[2]x[2]xi64>
  return %slice : vector<[2]xi64>
}

// -----

// CHECK-LABEL: @vector_extract_slice_i128
func.func @vector_extract_slice_i128(%row: index) -> vector<[1]xi128> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi128>, vector<[1]xi1>, i32) -> vector<[1]xi128>
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %slice = vector.extract %tile[%row] : vector<[1]xi128> from vector<[1]x[1]xi128>
  return %slice : vector<[1]xi128>
}

// -----

// CHECK-LABEL: @vector_extract_slice_f16
func.func @vector_extract_slice_f16(%row: index) -> vector<[8]xf16> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xf16>, vector<[8]xi1>, i32) -> vector<[8]xf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %slice = vector.extract %tile[%row] : vector<[8]xf16> from vector<[8]x[8]xf16>
  return %slice : vector<[8]xf16>
}

// -----

// CHECK-LABEL: @vector_extract_slice_bf16
func.func @vector_extract_slice_bf16(%row: index) -> vector<[8]xbf16> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xbf16>, vector<[8]xi1>, i32) -> vector<[8]xbf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %slice = vector.extract %tile[%row] : vector<[8]xbf16> from vector<[8]x[8]xbf16>
  return %slice : vector<[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_extract_slice_f32
func.func @vector_extract_slice_f32(%row: index) -> vector<[4]xf32> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %slice = vector.extract %tile[%row] : vector<[4]xf32> from vector<[4]x[4]xf32>
  return %slice : vector<[4]xf32>
}

// -----

// CHECK-LABEL: @vector_extract_slice_f64
func.func @vector_extract_slice_f64(%row: index) -> vector<[2]xf64> {
  // CHECK: %{{.*}} = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xf64>, vector<[2]xi1>, i32) -> vector<[2]xf64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %slice = vector.extract %tile[%row] : vector<[2]xf64> from vector<[2]x[2]xf64>
  return %slice : vector<[2]xf64>
}

// -----

// CHECK-LABEL: @vector_extract_element(
// CHECK-SAME:                          %[[ROW:.*]]: index,
// CHECK-SAME:                          %[[COL:.*]]: index)
func.func @vector_extract_element(%row: index, %col: index) -> i32 {
  // CHECK-NEXT: %[[PTRUE:.*]] = arith.constant dense<true> : vector<[4]xi1>
  // CHECK-NEXT: %[[ZERO_VEC:.*]] = arith.constant dense<0> : vector<[4]xi32>
  // CHECK-NEXT: %[[ROW_I32:.*]] = arith.index_cast %[[ROW]] : index to i32
  // CHECK-NEXT: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%[[ZERO_VEC]], %[[PTRUE]], %[[ROW_I32]]) <{tile_id = 0 : i32}> : (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>
  // CHECK-NEXT: %[[EL:.*]] = vector.extract %[[SLICE]]{{\[}}%[[COL]]] : i32 from vector<[4]xi32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  %el = vector.extract %tile[%row, %col] : i32 from vector<[4]x[4]xi32>
  return %el : i32
}

// -----

// CHECK-LABEL: @vector_extract_element_i8
func.func @vector_extract_element_i8(%row: index, %col: index) -> i8 {
  // CHECK: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[16]xi8>, vector<[16]xi1>, i32) -> vector<[16]xi8>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : i8 from vector<[16]xi8>
  %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  %el = vector.extract %tile[%row, %col] : i8 from vector<[16]x[16]xi8>
  return %el : i8
}

// -----

// CHECK-LABEL: @vector_extract_element_i16
func.func @vector_extract_element_i16(%row: index, %col: index) -> i16 {
  // CHECK: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xi16>, vector<[8]xi1>, i32) -> vector<[8]xi16>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : i16 from vector<[8]xi16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  %el = vector.extract %tile[%row, %col] : i16 from vector<[8]x[8]xi16>
  return %el : i16
}

// -----

// CHECK-LABEL: @vector_extract_element_i64
func.func @vector_extract_element_i64(%row: index, %col: index) -> i64 {
  // CHECK: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xi64>, vector<[2]xi1>, i32) -> vector<[2]xi64>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : i64 from vector<[2]xi64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  %el = vector.extract %tile[%row, %col] : i64 from vector<[2]x[2]xi64>
  return %el : i64
}

// -----

// CHECK-LABEL: @vector_extract_element_i128
func.func @vector_extract_element_i128(%row: index, %col: index) -> i128 {
  // CHECK: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[1]xi128>, vector<[1]xi1>, i32) -> vector<[1]xi128>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : i128 from vector<[1]xi128>
  %tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  %el = vector.extract %tile[%row, %col] : i128 from vector<[1]x[1]xi128>
  return %el : i128
}

// -----

// CHECK-LABEL: @vector_extract_element_f16
func.func @vector_extract_element_f16(%row: index, %col: index) -> f16 {
  // CHECK: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xf16>, vector<[8]xi1>, i32) -> vector<[8]xf16>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : f16 from vector<[8]xf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xf16>
  %el = vector.extract %tile[%row, %col] : f16 from vector<[8]x[8]xf16>
  return %el : f16
}

// -----

// CHECK-LABEL: @vector_extract_element_bf16
func.func @vector_extract_element_bf16(%row: index, %col: index) -> bf16 {
  // CHECK: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[8]xbf16>, vector<[8]xi1>, i32) -> vector<[8]xbf16>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : bf16 from vector<[8]xbf16>
  %tile = arm_sme.get_tile : vector<[8]x[8]xbf16>
  %el = vector.extract %tile[%row, %col] : bf16 from vector<[8]x[8]xbf16>
  return %el : bf16
}

// -----

// CHECK-LABEL: @vector_extract_element_f32
func.func @vector_extract_element_f32(%row: index, %col: index) -> f32 {
  // CHECK: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : f32 from vector<[4]xf32>
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %el = vector.extract %tile[%row, %col] : f32 from vector<[4]x[4]xf32>
  return %el : f32
}

// -----

// CHECK-LABEL: @vector_extract_element_f64
func.func @vector_extract_element_f64(%row: index, %col: index) -> f64 {
  // CHECK: %[[SLICE:.*]] = "arm_sme.intr.read.horiz"(%{{.*}}) <{tile_id = 0 : i32}> : (vector<[2]xf64>, vector<[2]xi1>, i32) -> vector<[2]xf64>
  // CHECK-NEXT: %{{.*}} = vector.extract %[[SLICE]]{{\[}}%{{.*}}] : f64 from vector<[2]xf64>
  %tile = arm_sme.get_tile : vector<[2]x[2]xf64>
  %el = vector.extract %tile[%row, %col] : f64 from vector<[2]x[2]xf64>
  return %el : f64
}
