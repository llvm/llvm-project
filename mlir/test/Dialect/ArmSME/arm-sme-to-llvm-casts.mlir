// RUN: mlir-opt %s -convert-arm-sme-to-scf -convert-vector-to-llvm="enable-arm-sme" -split-input-file | FileCheck %s

// This test verifies the temporary casts that are emitted when lowering to
// intrinsics to preserve data flow are correct. Canonicalization will remove
// these.

// CHECK-LABEL: @arm_sme_zero
// CHECK: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i8
// CHECK: arm_sme.intr.zero
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i8 to vector<[16]x[16]xi8>
// CHECK: scf.for
// CHECK:   %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[CAST_TILE_TO_VECTOR]] : vector<[16]x[16]xi8> to i8
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[CAST_VECTOR_TO_TILE]] : i8 to i32
// CHECK:   "arm_sme.intr.st1b.horiz"({{.*}}, {{.*}}, %[[TILE_ID_I32]], {{.*}}) : (vector<[16]xi1>, !llvm.ptr, i32, i32) -> ()
func.func @arm_sme_zero(%dest : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.zero : vector<[16]x[16]xi8>
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: @arm_sme_tile_load
// CHECK: %[[TILE_ID:.*]] = arm_sme.get_tile_id : i8
// CHECK: %[[CAST_TILE_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i8 to vector<[16]x[16]xi8>
// CHECK: scf.for
// CHECK:   %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[CAST_TILE_TO_VECTOR]] : vector<[16]x[16]xi8> to i8
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[CAST_VECTOR_TO_TILE]] : i8 to i32
// CHECK:   "arm_sme.intr.ld1b.horiz"({{.*}}, {{.*}}, %[[TILE_ID_I32]], {{.*}}) : (vector<[16]xi1>, !llvm.ptr, i32, i32) -> ()
// CHECK: }
// CHECK: return  %[[CAST_TILE_TO_VECTOR]] : vector<[16]x[16]xi8>
func.func @arm_sme_tile_load(%dest : memref<?x?xi8>) -> vector<[16]x[16]xi8> {
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %dest[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return %tile : vector<[16]x[16]xi8>
}

// -----

// CHECK-LABEL: @arm_sme_tile_store(
// CHECK-SAME:                      %[[TILE:.*]]: vector<[16]x[16]xi8>,
// CHECK: scf.for
// CHECK:   %[[CAST_VECTOR_TO_TILE:.*]] = arm_sme.cast_vector_to_tile %[[TILE]] : vector<[16]x[16]xi8> to i8
// CHECK:   %[[TILE_ID_I32:.*]] = arith.extui %[[CAST_VECTOR_TO_TILE]] : i8 to i32
// CHECK:   "arm_sme.intr.st1b.horiz"({{.*}}, {{.*}}, %[[TILE_ID_I32]], {{.*}}) : (vector<[16]xi1>, !llvm.ptr, i32, i32) -> ()
func.func @arm_sme_tile_store(%tile : vector<[16]x[16]xi8>, %dest : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}
