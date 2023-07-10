// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sme" -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: @transfer_write_2d_zero_i8
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xi8>)
// CHECK-NEXT: %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<?x?xi8> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[C255:.*]] = arith.constant 255 : i32
// CHECK-NEXT: "arm_sme.intr.zero"(%[[C255]]) : (i32) -> ()
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[MIN_ZA_VECTORS:.*]] = arith.constant 16 : index
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
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__bad_shape(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[8]x[8]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__bad_rank
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__bad_rank(%arg0 : memref<?x?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<[16]x[16]x[16]xi8>, memref<?x?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__non_memref_type
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__non_memref_type(%arg0 : tensor<?x?xi8>) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi8>
  %0 = vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__non_zero_value
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__non_zero_value(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1> : vector<[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__vec_unknown_defining_op
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__vec_unknown_defining_op(%arg0 : memref<?x?xi8>, %arg1 : vector<[16]x[16]xi8>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %arg1, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}
