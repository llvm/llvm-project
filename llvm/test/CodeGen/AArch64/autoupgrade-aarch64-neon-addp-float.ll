; RUN: opt -S < %s -mtriple=arm64 | FileCheck %s
declare <4 x float> @llvm.aarch64.neon.addp.v4f32(<4 x float>, <4 x float>)

; CHECK: call <4 x float> @llvm.aarch64.neon.faddp.v4f32
define <4 x float> @upgrade_aarch64_neon_addp_float(<4 x float> %a, <4 x float> %b) {
  %res = call <4 x float> @llvm.aarch64.neon.addp.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %res
}

