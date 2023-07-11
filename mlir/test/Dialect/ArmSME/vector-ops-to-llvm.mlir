// RUN: mlir-opt %s -convert-vector-to-arm-sme -convert-vector-to-llvm="enable-arm-sme" -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: @transfer_write_2d_zero_i8
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xi8>)
// CHECK-DAG: %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG: %[[C255:.*]] = arith.constant 255 : i32
// CHECK-DAG: "arm_sme.intr.zero"(%[[C255]]) : (i32) -> ()
// CHECK-DAG:  %[[TILE_ID:.*]] = arm_sme.get_tile_id : i8
// CHECK-DAG:  %[[CAST_TO_VECTOR:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i8 to vector<[16]x[16]xi8>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[MIN_ZA_VECTORS:.*]] = arith.constant 16 : index
// CHECK-NEXT: %[[VSCALE:.*]] = "llvm.intr.vscale"() : () -> i64
// CHECK-NEXT: %[[VSCALE_IDX:.*]] = builtin.unrealized_conversion_cast %[[VSCALE]] : i64 to index
// CHECK-NEXT: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[NUM_ZA_VECTORS:.*]] = arith.muli %[[MIN_ZA_VECTORS]], %[[VSCALE_IDX]] : index
// CHECK-NEXT: scf.for %[[VNUM:.*]] = %[[C0_0]] to %[[NUM_ZA_VECTORS]] step %[[C1]] {
// CHECK-NEXT:   %[[VNUM_I64:.*]] = arith.index_castui %[[VNUM]] : index to i64
// CHECK-NEXT:   %[[C0_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:   %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:   %[[OFF0:.*]] = llvm.mul %[[VNUM_I64]], %[[STRIDE0]]  : i64
// CHECK-NEXT:   %[[OFF1:.*]] = llvm.add %[[OFF0]], %[[C0_1]]  : i64
// CHECK-NEXT:   %[[GEP:.*]] = llvm.getelementptr %[[ALIGNED_BASE]]{{\[}}%[[OFF1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:   %[[VNUM_I32:.*]] = arith.index_castui %[[VNUM]] : index to i32
// CHECK-NEXT:   "arm_sme.intr.str"(%[[VNUM_I32]], %[[GEP]]) : (i32, !llvm.ptr) -> ()
func.func @transfer_write_2d_zero_i8(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

