// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!vec = vector<8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @shuffle_fma_with_rhs_as_even.index_to_f32(
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

// Groups FMAs with respect to even/odd indexed input operands.
// The vector.fma at %5 is moved along with its operands after %8.  
// CHECK-LABEL: @shuffle_fma_with_rhs_as_even.index_to_f32
// Odd-Indexed FMAs
// CHECK: %[[BCST0:.*]] = x86vector.avx.bcst_to_f32.packed %arg0
// CHECK: %[[ODD0:.*]]  = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2
// CHECK: %[[FMA_ODD0:.*]] = vector.fma %[[BCST0]], %[[ODD0]], %arg6
// CHECK: %[[BCST1:.*]] = x86vector.avx.bcst_to_f32.packed %arg3
// CHECK: %[[ODD1:.*]]  = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5
// CHECK: %[[FMA_ODD1:.*]] = vector.fma %[[BCST1]], %[[ODD1]], %arg6
// Even-Indexed FMAs
// CHECK: %[[BCST2:.*]] = x86vector.avx.bcst_to_f32.packed %arg4
// CHECK: %[[EVEN0:.*]] = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5
// CHECK: %[[FMA_EVEN0:.*]] = vector.fma %[[BCST2]], %[[EVEN0]], %[[FMA_ODD1]]
// CHECK: %[[BCST3:.*]] = x86vector.avx.bcst_to_f32.packed %arg1
// CHECK: %[[EVEN1:.*]] = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2
// CHECK: %[[FMA_EVEN1:.*]] = vector.fma %[[BCST3]], %[[EVEN1]], %[[FMA_ODD0]]

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

func.func @shuffle_fma_with_lhs_as_even.index_to_f32(
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

// The vector.fma at %5 is moved along with its operands after %8.
// CHECK-LABEL: @shuffle_fma_with_lhs_as_even.index_to_f32
// Odd-Indexed FMAs
// CHECK: %[[BCST0:.*]] = x86vector.avx.bcst_to_f32.packed %arg0
// CHECK: %[[ODD0:.*]]  = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2
// CHECK: %[[FMA_ODD0:.*]] = vector.fma %[[BCST0]], %[[ODD0]], %arg6
// CHECK: %[[BCST1:.*]] = x86vector.avx.bcst_to_f32.packed %arg3
// CHECK: %[[ODD1:.*]]  = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5
// CHECK: %[[FMA_ODD1:.*]] = vector.fma %[[BCST1]], %[[ODD1]], %arg6
// Even-Indexed FMAs
// CHECK: %[[BCST2:.*]] = x86vector.avx.bcst_to_f32.packed %arg4
// CHECK: %[[EVEN0:.*]] = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5
// CHECK: %[[FMA_EVEN0:.*]] = vector.fma %[[BCST2]], %[[EVEN0]], %[[FMA_ODD1]]
// CHECK: %[[EVEN1:.*]] = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2
// CHECK: %[[BCST3:.*]] = x86vector.avx.bcst_to_f32.packed %arg1
// CHECK: %[[FMA_EVEN1:.*]] = vector.fma %[[EVEN1]], %[[BCST3]], %[[FMA_ODD0]]

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
!vecOut = vector<1x8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @shuffle_fma_with_shape_cast(
  %arg0: !memrefA, %arg1: !memrefA, %arg2: !memrefB, %arg3: !memrefA,
  %arg4: !memrefA, %arg5: !memrefB, %arg6: !vec) -> !vecOut
{
  %0 = x86vector.avx.bcst_to_f32.packed %arg0 : !memrefA -> !vec
  %1 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2 : !memrefB -> !vec
  %2 = vector.fma %0, %1, %arg6 : !vec
  %3 = x86vector.avx.bcst_to_f32.packed %arg1 : !memrefA -> !vec
  %4 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2 : !memrefB -> !vec
  %5 = vector.fma %3, %4, %2 : !vec
  %res1 = vector.shape_cast %5 : !vec to !vecOut
  %6 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
  %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5 : !memrefB -> !vec
  %8 = vector.fma %6, %7, %arg6 : !vec
  %9 = x86vector.avx.bcst_to_f32.packed %arg4 : !memrefA -> !vec
  %10 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5 : !memrefB -> !vec
  %11 = vector.fma %9, %10, %8 : !vec
  %res2 = vector.shape_cast %11 : !vec to !vecOut
  %12 = arith.addf %res1, %res2 : !vecOut
  return %12 : !vecOut
}

// CHECK-LABEL: @shuffle_fma_with_shape_cast
// Odd-Indexed FMAs
// CHECK: %[[BCST0:.*]] = x86vector.avx.bcst_to_f32.packed %arg0
// CHECK: %[[ODD0:.*]]  = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2
// CHECK: %[[FMA_ODD0:.*]] = vector.fma %[[BCST0]], %[[ODD0]], %arg6
// CHECK: %[[BCST1:.*]] = x86vector.avx.bcst_to_f32.packed %arg3
// CHECK: %[[ODD1:.*]]  = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5
// CHECK: %[[FMA_ODD1:.*]] = vector.fma %[[BCST1]], %[[ODD1]], %arg6
// Even-Indexed FMAs
// CHECK: %[[BCST3:.*]] = x86vector.avx.bcst_to_f32.packed %arg4
// CHECK: %[[EVEN1:.*]] = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5
// CHECK: %[[FMA_EVEN1:.*]] = vector.fma %[[BCST3]], %[[EVEN1]], %[[FMA_ODD1]]
// CHECK: vector.shape_cast
// CHECK: %[[BCST2:.*]] = x86vector.avx.bcst_to_f32.packed %arg1
// CHECK: %[[EVEN0:.*]] = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2
// CHECK: %[[FMA_EVEN0:.*]] = vector.fma %[[BCST2]], %[[EVEN0]], %[[FMA_ODD0]]
// CHECK: vector.shape_cast

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

func.func @negative_fma_operand_has_multiple_consumer(
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

// The vector.fma at %5 uses %3 as its LHS operand, which has two consumers; therefore, 
// the rewrite is not applied.
// CHECK-LABEL: @negative_fma_operand_has_multiple_consumer
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
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

func.func @negative_fma_has_multiple_consumer(
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

// vector.fma at %5 has two uses; therefore no re-write applied.
// CHECK-LABEL: @negative_fma_has_multiple_consumer
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32

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

// vector.fma at %5 has its consumer in an another block (%12); therefore rewrite is not
// applied.
// CHECK-LABEL: @negative_no_shuffle_outside_block
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: vector.fma
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: vector.fma
// CHECK: scf.if
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
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
