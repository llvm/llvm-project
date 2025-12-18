// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!vec = vector<8xf32>
!memrefA = memref<1x1x1xbf16>
!memrefB = memref<1x8x2xbf16>

func.func @shuffle_vector_fma(
  %arg0: !memrefA, %arg1: !memrefA, %arg2: !memrefB, %arg3: !memrefA,
  %arg4: !memrefA, %arg5: !memrefB, %arg6: !vec) -> !vec
{
  %0 = x86vector.avx.bcst_to_f32.packed %arg0 : !memrefA -> !vec
  %1 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg2 : !memrefB -> !vec
  %2 = vector.fma %0, %1, %arg6 : !vec
  %3 = x86vector.avx.bcst_to_f32.packed %arg1 : !memrefA -> !vec
  %4 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg2 : !memrefB -> !vec
  %5 = vector.fma %3, %4, %2 : !vec
  %55 = vector.shape_cast %5 : vector<8xf32> to vector<1x8xf32>
  %6 = x86vector.avx.bcst_to_f32.packed %arg3 : !memrefA -> !vec
  %7 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %arg5 : !memrefB -> !vec
  %8 = vector.fma %6, %7, %arg6 : !vec
  %9 = x86vector.avx.bcst_to_f32.packed %arg4 : !memrefA -> !vec
  %10 = x86vector.avx.cvt.packed.even.indexed_to_f32 %arg5 : !memrefB -> !vec
  %11 = vector.fma %9, %10, %8 : !vec
  %13 = vector.shape_cast %55 : vector<1x8xf32> to vector<8xf32>
  %12 = vector.fma %13, %11, %arg6 : !vec
  return %12 : !vec
}

// CHECK-LABEL: @shuffle_vector_fma
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
