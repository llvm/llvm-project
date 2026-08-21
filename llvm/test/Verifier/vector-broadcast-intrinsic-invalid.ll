; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

; CHECK: vector_broadcast argument and result must have the same element type.
define <8 x i32> @mismatched_element_types(<2 x i64> %vec) {
  %result = call <8 x i32> @llvm.vector.broadcast.v8i32.v2i64(<2 x i64> %vec)
  ret <8 x i32> %result
}

; CHECK: vector_broadcast result element count must be a multiple of the argument element count.
define <8 x i32> @non_multiple_fixed(<3 x i32> %vec) {
  %result = call <8 x i32> @llvm.vector.broadcast.v8i32.v3i32(<3 x i32> %vec)
  ret <8 x i32> %result
}

; CHECK: vector_broadcast result element count must be a multiple of the argument element count.
define <vscale x 8 x i32> @non_multiple_scalable(<vscale x 3 x i32> %vec) {
  %result = call <vscale x 8 x i32> @llvm.vector.broadcast.nxv8i32.nxv3i32(<vscale x 3 x i32> %vec)
  ret <vscale x 8 x i32> %result
}

; CHECK: vector_broadcast cannot broadcast a scalable vector to a fixed-width vector.
define <8 x i32> @scalable_to_fixed(<vscale x 2 x i32> %vec) {
  %result = call <8 x i32> @llvm.vector.broadcast.v8i32.nxv2i32(<vscale x 2 x i32> %vec)
  ret <8 x i32> %result
}
