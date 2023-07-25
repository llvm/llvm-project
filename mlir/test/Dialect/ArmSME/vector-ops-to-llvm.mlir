// RUN: mlir-opt %s -convert-vector-to-arm-sme -convert-vector-to-llvm="enable-arm-sme" -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: @transfer_write_2d_zero_i8(
// CHECK-SAME:                             %[[ARG0:.*]]: memref<?x?xi8>)
// CHECK-DAG: %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG: %[[C255:.*]] = arith.constant 255 : i32
// CHECK-DAG: "arm_sme.intr.zero"(%[[C255]]) : (i32) -> ()
// CHECK-DAG:  %[[TILE_ID:.*]] = arm_sme.get_tile_id : i8
// CHECK-DAG:  %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i8 to vector<[16]x[16]xi8>
// CHECK-DAG:  %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[CAST_TILE_TO_VECTOR]] : vector<[16]x[16]xi8> to i8
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[MIN_SVL_B:.*]] = arith.constant 16 : index
// CHECK-NEXT: %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK-NEXT: %[[VSCALE_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[SVL_B:.*]] = arith.muli %[[MIN_SVL_B]], %[[VSCALE_IDX]] : index
// CHECK-NEXT: scf.for %[[TILE_SLICE:.*]] = %[[C0]] to %[[SVL_B]] step %[[C1]] {
// CHECK-NEXT:   %[[TILE_SLICE_I64:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[OFF0:.*]] = llvm.mul %[[TILE_SLICE_I64]], %[[STRIDE0]]  : i64
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFF0]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[TILE_SLICE_I32:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i32
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[16]xi1>
// CHECK-NEXT:   %[[TILE_ID_I32:.*]] = arith.extui %[[CAST_VECTOR_TO_TILE]] : i8 to i32
// CHECK-NEXT:   "arm_sme.intr.st1b.horiz"(%[[PTRUE_ALL]], %[[GEP]], %[[TILE_ID_I32]], %[[TILE_SLICE_I32]]) : (vector<[16]xi1>, !llvm.ptr, i32, i32) -> ()
func.func @transfer_write_2d_zero_i8(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: @vector_load_i8(
// CHECK-SAME:                  %[[ARG0:.*]]: memref<?x?xi8>)
// CHECK-NEXT: %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i8
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[MIN_SVL_B:.*]] = arith.constant 16 : index
// CHECK-NEXT: %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK-NEXT: %[[VSCALE_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK-NEXT: %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[SVL_B:.*]] = arith.muli %[[MIN_SVL_B]], %[[VSCALE_IDX]] : index
// CHECK-NEXT: scf.for %[[TILE_SLICE:.*]] = %[[C0_1]] to %[[SVL_B]] step %[[C1]] {
// CHECK-NEXT:   %[[TILE_SLICE_I64:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[OFF0:.*]] = llvm.mul %[[TILE_SLICE_I64]], %[[STRIDE0]]  : i64
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFF0]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[TILE_SLICE_I32:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i32
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[16]xi1>
// CHECK-NEXT:   %[[TILE_ID_I32:.*]] = arith.extui %[[TILE_ID]] : i8 to i32
// CHECK-NEXT:   "arm_sme.intr.ld1b.horiz"(%[[PTRUE_ALL]], %[[GEP]], %[[TILE_ID_I32]], %[[TILE_SLICE_I32]]) : (vector<[16]xi1>, !llvm.ptr, i32, i32) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i8 to vector<[16]x[16]xi8>
// CHECK-NEXT: return  %[[CAST_TILE_TO_VECTOR]] : vector<[16]x[16]xi8>
func.func @vector_load_i8(%arg0 : memref<?x?xi8>) -> vector<[16]x[16]xi8> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return %tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @vector_load_i8_from_rank_1_memref(
// CHECK-SAME:                                     %[[ARG0:.*]]: memref<?xi8>)
// CHECK-NEXT: %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?xi8> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i8
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[MIN_SVL_B:.*]] = arith.constant 16 : index
// CHECK-NEXT: %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK-NEXT: %[[VSCALE_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK-NEXT: %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[SVL_B:.*]] = arith.muli %[[MIN_SVL_B]], %[[VSCALE_IDX]] : index
// CHECK-NEXT: scf.for %[[TILE_SLICE:.*]] = %[[C0_1]] to %[[SVL_B]] step %[[C1]] {
// CHECK-NEXT:   %[[TILE_SLICE_I64:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i64
// CHECK-NEXT:   %[[SVL_B_I64:.*]] = arith.index_castui %[[SVL_B]] : index to i64
// CHECK-NEXT:   %[[TILE_SLICE_IDX:.*]] = arith.muli %[[TILE_SLICE_I64]], %[[SVL_B_I64]] : i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[TILE_SLICE_IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[TILE_SLICE_I32:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i32
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[16]xi1>
// CHECK-NEXT:   %[[TILE_ID_I32:.*]] = arith.extui %[[TILE_ID]] : i8 to i32
// CHECK-NEXT:   "arm_sme.intr.ld1b.horiz"(%[[PTRUE_ALL]], %[[GEP]], %[[TILE_ID_I32]], %[[TILE_SLICE_I32]]) : (vector<[16]xi1>, !llvm.ptr, i32, i32) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i8 to vector<[16]x[16]xi8>
// CHECK-NEXT: return  %[[CAST_TILE_TO_VECTOR]] : vector<[16]x[16]xi8>
func.func @vector_load_i8_from_rank_1_memref(%arg0 : memref<?xi8>) -> vector<[16]x[16]xi8> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0] : memref<?xi8>, vector<[16]x[16]xi8>
  return %tile : vector<[16]x[16]xi8>
}


// -----

// CHECK-LABEL: @vector_load_i16(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi16>)
// CHECK: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i16
// CHECK: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK: %[[SVL_H:.*]] = arith.muli %[[MIN_SVL_H]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[TILE_ID]] : i16 to i32
// CHECK:   arm_sme.intr.ld1h.horiz
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i16 to vector<[8]x[8]xi16>
func.func @vector_load_i16(%arg0 : memref<?x?xi16>) -> vector<[8]x[8]xi16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return %tile : vector<[8]x[8]xi16>
}

// -----

// CHECK-LABEL: @vector_load_i32(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi32>)
// CHECK:     %[[TILE_ID:.*]] = arm_sme.get_tile_id : i32
// CHECK:     %[[MIN_SVL_S:.*]] = arith.constant 4 : index
// CHECK:     %[[SVL_S:.*]] = arith.muli %[[MIN_SVL_S]], %{{.*}} : index
// CHECK-NOT:   arith.extui %[[TILE_ID]]
// CHECK-NOT:   arith.trunci %[[TILE_ID]]
// CHECK:       arm_sme.intr.ld1w.horiz
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i32 to vector<[4]x[4]xi32>
func.func @vector_load_i32(%arg0 : memref<?x?xi32>) -> vector<[4]x[4]xi32> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return %tile : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @vector_load_i64(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi64>)
// CHECK: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i64
// CHECK: %[[MIN_SVL_D:.*]] = arith.constant 2 : index
// CHECK: %[[SVL_D:.*]] = arith.muli %[[MIN_SVL_D]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.trunci %[[TILE_ID]] : i64 to i32
// CHECK:   arm_sme.intr.ld1d.horiz
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i64 to vector<[2]x[2]xi64>
func.func @vector_load_i64(%arg0 : memref<?x?xi64>) -> vector<[2]x[2]xi64> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return %tile : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @vector_load_f16(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf16>)
// CHECK: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i16
// CHECK: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK: %[[SVL_H:.*]] = arith.muli %[[MIN_SVL_H]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[TILE_ID]] : i16 to i32
// CHECK:   arm_sme.intr.ld1h.horiz
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i16 to vector<[8]x[8]xf16>
func.func @vector_load_f16(%arg0 : memref<?x?xf16>) -> vector<[8]x[8]xf16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return %tile : vector<[8]x[8]xf16>
}

// -----

// CHECK-LABEL: @vector_load_bf16(
// CHECK-SAME:                    %[[ARG0:.*]]: memref<?x?xbf16>)
// CHECK: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i16
// CHECK: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK: %[[SVL_H:.*]] = arith.muli %[[MIN_SVL_H]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[TILE_ID]] : i16 to i32
// CHECK:   arm_sme.intr.ld1h.horiz
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i16 to vector<[8]x[8]xbf16>
func.func @vector_load_bf16(%arg0 : memref<?x?xbf16>) -> vector<[8]x[8]xbf16> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return %tile : vector<[8]x[8]xbf16>
}

// -----

// CHECK-LABEL: @vector_load_f32(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf32>)
// CHECK:     %[[TILE_ID:.*]] = arm_sme.get_tile_id : i32
// CHECK:     %[[MIN_SVL_S:.*]] = arith.constant 4 : index
// CHECK:     %[[SVL_S:.*]] = arith.muli %[[MIN_SVL_S]], %{{.*}} : index
// CHECK-NOT:   arith.extui %[[TILE_ID]]
// CHECK-NOT:   arith.trunci %[[TILE_ID]]
// CHECK:       arm_sme.intr.ld1w.horiz
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i32 to vector<[4]x[4]xf32>
func.func @vector_load_f32(%arg0 : memref<?x?xf32>) -> vector<[4]x[4]xf32> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return %tile : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @vector_load_f64(
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf64>)
// CHECK: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i64
// CHECK: %[[MIN_SVL_D:.*]] = arith.constant 2 : index
// CHECK: %[[SVL_D:.*]] = arith.muli %[[MIN_SVL_D]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.trunci %[[TILE_ID]] : i64 to i32
// CHECK:   arm_sme.intr.ld1d.horiz
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i64 to vector<[2]x[2]xf64>
func.func @vector_load_f64(%arg0 : memref<?x?xf64>) -> vector<[2]x[2]xf64> {
  %c0 = arith.constant 0 : index
  %tile = vector.load %arg0[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return %tile : vector<[2]x[2]xf64>
}

// -----

// CHECK-LABEL: @vector_store_i8(
// CHECK-SAME:                   %[[TILE:.*]]: vector<[16]x[16]xi8>,
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi8>)
// CHECK-NEXT: %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[16]x[16]xi8> to i8
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[MIN_SVL_B:.*]] = arith.constant 16 : index
// CHECK-NEXT: %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK-NEXT: %[[VSCALE_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK-NEXT: %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[SVL_B:.*]] = arith.muli %[[MIN_SVL_B]], %[[VSCALE_IDX]] : index
// CHECK-NEXT: scf.for %[[TILE_SLICE:.*]] = %[[C0_1]] to %[[SVL_B]] step %[[C1]] {
// CHECK-NEXT:   %[[TILE_SLICE_I64:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[OFF0:.*]] = llvm.mul %[[TILE_SLICE_I64]], %[[STRIDE0]]  : i64
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFF0]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[TILE_SLICE_I32:.*]] = arith.index_castui %[[TILE_SLICE]] : index to i32
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[PTRUE_ALL:.*]] = arith.constant dense<true> : vector<[16]xi1>
// CHECK-NEXT:   %[[TILE_ID_I32:.*]] = arith.extui %[[CAST_VECTOR_TO_TILE]] : i8 to i32
// CHECK-NEXT:   "arm_sme.intr.st1b.horiz"(%[[PTRUE_ALL]], %[[GEP]], %[[TILE_ID_I32]], %[[TILE_SLICE_I32]]) : (vector<[16]xi1>, !llvm.ptr, i32, i32) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @vector_store_i8(%tile : vector<[16]x[16]xi8>, %arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: @vector_store_i16(
// CHECK-SAME:                   %[[TILE:.*]]: vector<[8]x[8]xi16>,
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi16>)
// CHECK: %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[8]x[8]xi16> to i16
// CHECK: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK: %[[SVL_H:.*]] = arith.muli %[[MIN_SVL_H]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[CAST_VECTOR_TO_TILE]] : i16 to i32
// CHECK:   arm_sme.intr.st1h.horiz
func.func @vector_store_i16(%tile : vector<[8]x[8]xi16>, %arg0 : memref<?x?xi16>) {
  %c0 = arith.constant 0 : index
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return
}

// -----

// CHECK-LABEL: @vector_store_i32(
// CHECK-SAME:                   %[[TILE:.*]]: vector<[4]x[4]xi32>,
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi32>)
// CHECK:     %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[4]x[4]xi32> to i32
// CHECK:     %[[MIN_SVL_S:.*]] = arith.constant 4 : index
// CHECK:     %[[SVL_S:.*]] = arith.muli %[[MIN_SVL_S]], %{{.*}} : index
// CHECK-NOT:   arith.extui %[[CAST_VECTOR_TO_TILE]]
// CHECK-NOT:   arith.trunci %[[CAST_VECTOR_TO_TILE]]
// CHECK:       arm_sme.intr.st1w.horiz
func.func @vector_store_i32(%tile : vector<[4]x[4]xi32>, %arg0 : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @vector_store_i64(
// CHECK-SAME:                   %[[TILE:.*]]: vector<[2]x[2]xi64>,
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xi64>)
// CHECK: %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[2]x[2]xi64> to i64
// CHECK: %[[MIN_SVL_D:.*]] = arith.constant 2 : index
// CHECK: %[[SVL_D:.*]] = arith.muli %[[MIN_SVL_D]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.trunci %[[CAST_VECTOR_TO_TILE]] : i64 to i32
// CHECK:   arm_sme.intr.st1d.horiz
func.func @vector_store_i64(%tile : vector<[2]x[2]xi64>, %arg0 : memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return
}

// -----

// CHECK-LABEL: @vector_store_f16(
// CHECK-SAME:                   %[[TILE:.*]]: vector<[8]x[8]xf16>,
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf16>)
// CHECK: %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[8]x[8]xf16> to i16
// CHECK: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK: %[[SVL_H:.*]] = arith.muli %[[MIN_SVL_H]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[CAST_VECTOR_TO_TILE]] : i16 to i32
// CHECK:   arm_sme.intr.st1h.horiz
func.func @vector_store_f16(%tile : vector<[8]x[8]xf16>, %arg0 : memref<?x?xf16>) {
  %c0 = arith.constant 0 : index
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return
}

// -----

// CHECK-LABEL: @vector_store_bf16(
// CHECK-SAME:                   %[[TILE:.*]]: vector<[8]x[8]xbf16>,
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xbf16>)
// CHECK: %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[8]x[8]xbf16> to i16
// CHECK: %[[MIN_SVL_H:.*]] = arith.constant 8 : index
// CHECK: %[[SVL_H:.*]] = arith.muli %[[MIN_SVL_H]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[CAST_VECTOR_TO_TILE]] : i16 to i32
// CHECK:   arm_sme.intr.st1h.horiz
func.func @vector_store_bf16(%tile : vector<[8]x[8]xbf16>, %arg0 : memref<?x?xbf16>) {
  %c0 = arith.constant 0 : index
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return
}
// -----

// CHECK-LABEL: @vector_store_f32(
// CHECK-SAME:                   %[[TILE:.*]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf32>)
// CHECK:     %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[4]x[4]xf32> to i32
// CHECK:     %[[MIN_SVL_S:.*]] = arith.constant 4 : index
// CHECK:     %[[SVL_S:.*]] = arith.muli %[[MIN_SVL_S]], %{{.*}} : index
// CHECK-NOT:   arith.extui %[[CAST_VECTOR_TO_TILE]]
// CHECK-NOT:   arith.trunci %[[CAST_VECTOR_TO_TILE]]
// CHECK:       arm_sme.intr.st1w.horiz
func.func @vector_store_f32(%tile : vector<[4]x[4]xf32>, %arg0 : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

// CHECK-LABEL: @vector_store_f64(
// CHECK-SAME:                   %[[TILE:.*]]: vector<[2]x[2]xf64>,
// CHECK-SAME:                   %[[ARG0:.*]]: memref<?x?xf64>)
// CHECK: %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[2]x[2]xf64> to i64
// CHECK: %[[MIN_SVL_D:.*]] = arith.constant 2 : index
// CHECK: %[[SVL_D:.*]] = arith.muli %[[MIN_SVL_D]], %{{.*}} : index
// CHECK:   %[[TILE_ID_I32:.*]] = arith.trunci %[[CAST_VECTOR_TO_TILE]] : i64 to i32
// CHECK:   arm_sme.intr.st1d.horiz
func.func @vector_store_f64(%tile : vector<[2]x[2]xf64>, %arg0 : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  vector.store %tile, %arg0[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}
