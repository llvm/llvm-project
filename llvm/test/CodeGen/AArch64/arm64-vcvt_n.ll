; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <2 x float> @cvtf32fxpu(<2 x i32> %a) nounwind readnone ssp {
; CHECK-LABEL: cvtf32fxpu:
; CHECK: ucvtf.2s	v0, v0, #9
; CHECK: ret
  %vcvt_n1 = tail call <2 x float> @llvm.aarch64.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32> %a, i32 9)
  ret <2 x float> %vcvt_n1
}

define <2 x float> @cvtf32fxps(<2 x i32> %a) nounwind readnone ssp {
; CHECK-LABEL: cvtf32fxps:
; CHECK: scvtf.2s	v0, v0, #12
; CHECK: ret
  %vcvt_n1 = tail call <2 x float> @llvm.aarch64.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32> %a, i32 12)
  ret <2 x float> %vcvt_n1
}

define <4 x float> @cvtqf32fxpu(<4 x i32> %a) nounwind readnone ssp {
; CHECK-LABEL: cvtqf32fxpu:
; CHECK: ucvtf.4s	v0, v0, #18
; CHECK: ret
  %vcvt_n1 = tail call <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> %a, i32 18)
  ret <4 x float> %vcvt_n1
}

define <4 x float> @cvtqf32fxps(<4 x i32> %a) nounwind readnone ssp {
; CHECK-LABEL: cvtqf32fxps:
; CHECK: scvtf.4s	v0, v0, #30
; CHECK: ret
  %vcvt_n1 = tail call <4 x float> @llvm.aarch64.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32> %a, i32 30)
  ret <4 x float> %vcvt_n1
}
define <2 x double> @f1(<2 x i64> %a) nounwind readnone ssp {
  %vcvt_n1 = tail call <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64> %a, i32 12)
  ret <2 x double> %vcvt_n1
}

define <2 x double> @f2(<2 x i64> %a) nounwind readnone ssp {
  %vcvt_n1 = tail call <2 x double> @llvm.aarch64.neon.vcvtfxs2fp.v2f64.v2i64(<2 x i64> %a, i32 9)
  ret <2 x double> %vcvt_n1
}

declare <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32>, i32) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32>, i32) nounwind readnone
declare <2 x float> @llvm.aarch64.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32>, i32) nounwind readnone
declare <2 x float> @llvm.aarch64.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32>, i32) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64>, i32) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.vcvtfxs2fp.v2f64.v2i64(<2 x i64>, i32) nounwind readnone
