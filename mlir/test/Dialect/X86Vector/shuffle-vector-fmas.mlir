// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!vec = vector<8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @shuffle_fma_lhs_even_index(
  %arg0: !memrefA, %arg1: !memrefA, %arg2: !memrefB, %arg3: !memrefA,
  %arg4: !memrefA, %arg5: !memrefB, %arg6: !vec) -> !vec
{
  %0 = x86vector.avx.bcst_to_f32.packed %arg0 : !memrefA -> !vec
  %1 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2 : !memrefB -> !vec
  %2 = vector.fma %0, %1, %arg6 : !vec
  %3 = x86vector.avx.bcst_to_f32.packed %arg1 : !memrefA -> !vec
  %4 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2 : !memrefB -> !vec
  %5 = vector.fma %3, %4, %2 : !vec
  %6 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
  %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5 : !memrefB -> !vec
  %8 = vector.fma %6, %7, %arg6 : !vec
  %9 = x86vector.avx.bcst_to_f32.packed %arg4 : !memrefA -> !vec
  %10 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5 : !memrefB -> !vec
  %11 = vector.fma %9, %10, %8 : !vec
  %12 = vector.fma %5, %11, %arg6 : !vec
  return %12 : !vec
}

// CHECK-LABEL: @shuffle_fma_lhs_even_index
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.shuffle_vector_fma_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vec = vector<8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @shuffle_fma_rhs_even_index(
  %arg0: !memrefA, %arg1: !memrefA, %arg2: !memrefB, %arg3: !memrefA,
  %arg4: !memrefA, %arg5: !memrefB, %arg6: !vec) -> !vec
{
  %0 = x86vector.avx.bcst_to_f32.packed %arg0 : !memrefA -> !vec
  %1 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2 : !memrefB -> !vec
  %2 = vector.fma %0, %1, %arg6 : !vec
  %3 = x86vector.avx.bcst_to_f32.packed %arg1 : !memrefA -> !vec
  %4 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2 : !memrefB -> !vec
  %5 = vector.fma %4, %3, %2 : !vec
  %6 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
  %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5 : !memrefB -> !vec
  %8 = vector.fma %6, %7, %arg6 : !vec
  %9 = x86vector.avx.bcst_to_f32.packed %arg4 : !memrefA -> !vec
  %10 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5 : !memrefB -> !vec
  %11 = vector.fma %9, %10, %8 : !vec
  %12 = vector.fma %5, %11, %arg6 : !vec
  return %12 : !vec
}

// CHECK-LABEL: @shuffle_fma_rhs_even_index
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.shuffle_vector_fma_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vec = vector<8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @negative_fma_lhs_multiple_consumer(
  %arg0: !memrefA, %arg1: !memrefA, %arg2: !memrefB,
  %arg3: !memrefA, %arg4: !memrefB, %arg5: !vec) -> !vec
{
  %0 = x86vector.avx.bcst_to_f32.packed %arg0 : !memrefA -> !vec
  %1 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2 : !memrefB -> !vec
  %2 = vector.fma %0, %1, %arg5 : !vec
  %3 = x86vector.avx.bcst_to_f32.packed %arg1 : !memrefA -> !vec
  %4 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2 : !memrefB -> !vec
  %5 = vector.fma %3, %4, %2 : !vec
  %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg4 : !memrefB -> !vec
  %8 = vector.fma %3, %7, %arg5 : !vec
  %9 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
  %10 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg4 : !memrefB -> !vec
  %11 = vector.fma %9, %10, %8 : !vec
  %12 = vector.fma %5, %11, %arg5 : !vec
  return %12 : !vec
}

// CHECK-LABEL: @negative_fma_lhs_multiple_consumer
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.shuffle_vector_fma_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vec = vector<8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @negative_fma_rhs_multiple_consumer(
  %arg0: !memrefA, %arg1: !memrefA, %arg2: !memrefB, %arg3: !memrefA,
  %arg4: !memrefA, %arg5: !memrefB, %arg6: !vec) -> !vec
{
  %0 = x86vector.avx.bcst_to_f32.packed %arg0 : !memrefA -> !vec
  %1 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2 : !memrefB -> !vec
  %2 = vector.fma %0, %1, %arg6 : !vec
  %3 = x86vector.avx.bcst_to_f32.packed %arg1 : !memrefA -> !vec
  %4 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2 : !memrefB -> !vec
  %5 = vector.fma %3, %4, %2 : !vec
  %6 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
  %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5 : !memrefB -> !vec
  %8 = vector.fma %6, %7, %arg6 : !vec
  %9 = x86vector.avx.bcst_to_f32.packed %arg4 : !memrefA -> !vec
  %10 = vector.fma %9, %4, %8 : !vec
  %11 = vector.fma %5, %10, %arg6 : !vec
  return %11 : !vec
}

// CHECK-LABEL: @negative_fma_rhs_multiple_consumer
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.shuffle_vector_fma_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vec = vector<8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @negative_fma_multiple_consumer(
  %arg0: !memrefA, %arg1: !memrefA, %arg2: !memrefB, %arg3: !memrefA,
  %arg4: !memrefA, %arg5: !memrefB, %arg6: !vec) -> !vec
{
  %0 = x86vector.avx.bcst_to_f32.packed %arg0 : !memrefA -> !vec
  %1 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2 : !memrefB -> !vec
  %2 = vector.fma %0, %1, %arg6 : !vec
  %3 = x86vector.avx.bcst_to_f32.packed %arg1 : !memrefA -> !vec
  %4 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2 : !memrefB -> !vec
  %5 = vector.fma %3, %4, %2 : !vec
  %6 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
  %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5 : !memrefB -> !vec
  %8 = vector.fma %6, %7, %5 : !vec
  %9 = x86vector.avx.bcst_to_f32.packed %arg4 : !memrefA -> !vec
  %10 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5 : !memrefB -> !vec
  %11 = vector.fma %9, %10, %8 : !vec
  %12 = vector.fma %5, %11, %arg6 : !vec
  return %12 : !vec
}

// CHECK-LABEL: @negative_fma_multiple_consumer
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.shuffle_vector_fma_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----
!vec = vector<8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @negative_no_shuffle_outside_block(
  %arg0: !memrefA, %arg1: !memrefA, %arg2: !memrefB, %arg3: !memrefA,
  %arg4: !memrefA, %arg5: !memrefB, %arg6: !vec, %arg7: i1) -> !vec
{
  %0 = x86vector.avx.bcst_to_f32.packed %arg0 : !memrefA -> !vec
  %1 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2 : !memrefB -> !vec
  %2 = vector.fma %0, %1, %arg6 : !vec
  %3 = x86vector.avx.bcst_to_f32.packed %arg1 : !memrefA -> !vec
  %4 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2 : !memrefB -> !vec
  %5 = vector.fma %3, %4, %2 : !vec

  %loop = scf.if %arg7 -> (vector<8xf32>) {
    %6 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
    %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5 : !memrefB -> !vec
    %8 = vector.fma %6, %7, %arg6 : !vec
    %9 = x86vector.avx.bcst_to_f32.packed %arg4 : !memrefA -> !vec
    %10 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5 : !memrefB -> !vec
    %11 = vector.fma %9, %10, %8 : !vec
    %12 = vector.fma %5, %11, %arg6 : !vec
    scf.yield %12 : vector<8xf32>
  } else {
    %6 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
    %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5 : !memrefB -> !vec
    %8 = vector.fma %6, %7, %arg6 : !vec
    %9 = x86vector.avx.bcst_to_f32.packed %arg4 : !memrefA -> !vec
    %10 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5 : !memrefB -> !vec
    %11 = vector.fma %9, %10, %8 : !vec
    %12 = vector.fma %5, %11, %arg6 : !vec
    scf.yield %12 : vector<8xf32>
  }

  return %loop : !vec
}

// CHECK-LABEL: @negative_no_shuffle_outside_block
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: scf.if
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.bcst_to_f32.packed
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.shuffle_vector_fma_ops
    } : !transform.any_op
    transform.yield
  }
}
