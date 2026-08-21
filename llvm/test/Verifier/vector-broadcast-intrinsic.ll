; RUN: opt -passes=verify -disable-output < %s

define <8 x i32> @fixed_to_fixed(<2 x i32> %vec) {
  %result = call <8 x i32> @llvm.vector.broadcast.v8i32.v2i32(<2 x i32> %vec)
  ret <8 x i32> %result
}

define <vscale x 8 x i32> @scalable_to_scalable(<vscale x 2 x i32> %vec) {
  %result = call <vscale x 8 x i32> @llvm.vector.broadcast.nxv8i32.nxv2i32(<vscale x 2 x i32> %vec)
  ret <vscale x 8 x i32> %result
}

define <vscale x 2 x i32> @fixed_to_scalable_with_vscale_range(<4 x i32> %vec) vscale_range(2, 8) {
  %result = call <vscale x 2 x i32> @llvm.vector.broadcast.nxv2i32.v4i32(<4 x i32> %vec)
  ret <vscale x 2 x i32> %result
}

define <vscale x 2 x i32> @fixed_to_scalable_without_vscale_range(<4 x i32> %vec) {
  %result = call <vscale x 2 x i32> @llvm.vector.broadcast.nxv2i32.v4i32(<4 x i32> %vec)
  ret <vscale x 2 x i32> %result
}

define <vscale x 2 x i32> @fixed_to_scalable_without_sufficient_vscale_range(<8 x i32> %vec) vscale_range(2, 8) {
  %result = call <vscale x 2 x i32> @llvm.vector.broadcast.nxv2i32.v8i32(<8 x i32> %vec)
  ret <vscale x 2 x i32> %result
}
