; RUN: not opt -S < %s 2>&1 | FileCheck %s

; Test that AArch64 NEON intrinsics with incorrect vector element types
; are rejected with a proper error instead of causing assertion failures.
; This is a regression test for GitHub issue #176847.

; CHECK: invalid intrinsic signature
define <vscale x 4 x float> @test_raddhn_float_instead_of_int(<vscale x 4 x float> %arg0) {
  %i = call <vscale x 4 x float> @llvm.aarch64.neon.raddhn.v4f32(<vscale x 4 x float> %arg0, i32 0)
  ret <vscale x 4 x float> %i
}

; CHECK: invalid intrinsic signature
define <8 x half> @test_sqdmull_half_instead_of_int(<8 x half> %arg0, <8 x half> %arg1) {
  %i = call <8 x half> @llvm.aarch64.neon.sqdmull.v8f16(<8 x half> %arg0, <8 x half> %arg1)
  ret <8 x half> %i
}
