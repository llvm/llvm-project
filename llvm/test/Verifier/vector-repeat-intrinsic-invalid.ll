; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

; CHECK: vector_repeat argument and result must have the same element type.
define <vscale x 8 x i32> @mismatched_element_types(<8 x i64> %vec) {
  %result = call <vscale x 8 x i32> @llvm.vector.repeat.nxv8i32.v8i64(<8 x i64> %vec)
  ret <vscale x 8 x i32> %result
}

; CHECK: vector_repeat result must be a scalable vector.
define <8 x i32> @fixed_to_fixed(<8 x i32> %vec) {
  %result = call <8 x i32> @llvm.vector.repeat.v8i32(<8 x i32> %vec)
  ret <8 x i32> %result
}

; CHECK: vector_repeat argument must be a fixed-length vector.
define <vscale x 8 x i32> @scalable_to_scalable(<vscale x 8 x i32> %vec) {
  %result = call <vscale x 8 x i32> @llvm.vector.repeat.nxv8i32(<vscale x 8 x i32> %vec)
  ret <vscale x 8 x i32> %result
}

; CHECK: vector_repeat argument and result must have the same minimum element count.
define <vscale x 8 x i32> @mismatched_minimum_element_count(<4 x i32> %vec) {
  %result = call <vscale x 8 x i32> @llvm.vector.repeat.nxv8i32.v4i32(<4 x i32> %vec)
  ret <vscale x 8 x i32> %result
}
