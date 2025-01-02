// Note: We run CSE here to make the pattern matching more direct.

// RUN: mlir-opt %s -test-lower-to-llvm -cse -canonicalize | FileCheck %s

// RUN: mlir-opt %s \
// RUN:   -transform-preload-library=transform-library-paths=%p/../Transform/include/Library/lower-to-llvm.mlir \
// RUN:   -transform-interpreter="entry-point=entry_point" \
// RUN:   -test-transform-dialect-erase-schedule \
// RUN:   -cse -canonicalize \
// RUN: | FileCheck %s

// Check that we properly lower to llvm memref operations that require to be
// expanded first, like `memref.subview`.
func.func @subview(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>>, %arg0 : index, %arg1 : index, %arg2 : index)
-> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK-LABEL: @subview
  // CHECK-SAME: %[[BASE:[^:]*]]: !llvm.ptr
  // CHECK-SAME: %[[BASE_ALIGNED:[^:]*]]: !llvm.ptr,
  // CHECK-SAME: %[[BASE_OFFSET:[^:]*]]: i64,
  // CHECK-SAME: %[[BASE_STRIDE0:[^:]*]]: i64,
  // CHECK-SAME: %[[BASE_STRIDE1:[^:]*]]: i64,
  // CHECK-SAME: %[[BASE_SIZE0:[^:]*]]: i64,
  // CHECK-SAME: %[[BASE_SIZE1:[^:]*]]: i64,
  // CHECK-SAME: %[[ARG0:[^:]*]]: i64,
  // CHECK-SAME: %[[ARG1:[^:]*]]: i64,
  // CHECK-SAME: %[[ARG2:[^:]*]]: i64)
  // CHECK-SAME: -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>

  // CHECK-DAG: %[[STRIDE0:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK-DAG: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] overflow<nsw> : i64
  // CHECK-DAG: %[[OFF2:.*]] = llvm.add %[[DESCSTRIDE0]], %[[ARG1]] : i64
  // CHECK-DAG: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

  // Base address and algined address.
  // CHECK-DAG: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

  // Offset.
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // Size 0.
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC2]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // Stride 0 == 4 * %arg0.
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC3]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // Size 1.
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC4]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // Stride 1 == 1 * %arg1.
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
  to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}

module @named_inclusion_in_named attributes { transform.with_named_sequence } {
  transform.named_sequence private @lower_to_llvm(!transform.any_op {transform.readonly}) -> !transform.any_op

  transform.named_sequence @entry_point(
    %toplevel_module : !transform.any_op {transform.readonly}) {
    transform.include @lower_to_llvm failures(suppress) (%toplevel_module) 
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
