; RUN: opt -passes=verify -disable-output < %s

define <vscale x 4 x i32> @fixed_to_scalable(<4 x i32> %vec) {
  %result = call <vscale x 4 x i32> @llvm.vector.repeat.nxv4i32.v4i32(<4 x i32> %vec)
  ret <vscale x 4 x i32> %result
}
