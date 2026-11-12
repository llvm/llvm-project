; RUN: llc -mtriple aarch64-none-linux-gnu -mattr=+fp16fml < %s | FileCheck %s

declare <2 x float> @llvm.aarch64.neon.fmlal.v2f32.v4f16(<2 x float>, <4 x half>, <4 x half>)
declare <2 x float> @llvm.aarch64.neon.fmlsl.v2f32.v4f16(<2 x float>, <4 x half>, <4 x half>)
declare <2 x float> @llvm.aarch64.neon.fmlal2.v2f32.v4f16(<2 x float>, <4 x half>, <4 x half>)
declare <2 x float> @llvm.aarch64.neon.fmlsl2.v2f32.v4f16(<2 x float>, <4 x half>, <4 x half>)
declare <4 x float> @llvm.aarch64.neon.fmlal.v4f32.v8f16(<4 x float>, <8 x half>, <8 x half>)
declare <4 x float> @llvm.aarch64.neon.fmlsl.v4f32.v8f16(<4 x float>, <8 x half>, <8 x half>)
declare <4 x float> @llvm.aarch64.neon.fmlal2.v4f32.v8f16(<4 x float>, <8 x half>, <8 x half>)
declare <4 x float> @llvm.aarch64.neon.fmlsl2.v4f32.v8f16(<4 x float>, <8 x half>, <8 x half>)

define <2 x float> @test_vfmlal_low_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c) #0 {
entry:
; CHECK-LABEL: test_vfmlal_low_u32:
; CHECK: fmlal   v0.2s, v1.2h, v2.2h
  %vfmlal_low2.i = call <2 x float> @llvm.aarch64.neon.fmlal.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> %c) #2
  ret <2 x float> %vfmlal_low2.i
}

define <2 x float> @test_vfmlsl_low_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c) #0 {
entry:
; CHECK-LABEL: test_vfmlsl_low_u32:
; CHECK: fmlsl   v0.2s, v1.2h, v2.2h
  %vfmlsl_low2.i = call <2 x float> @llvm.aarch64.neon.fmlsl.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> %c) #2
  ret <2 x float> %vfmlsl_low2.i
}

define <2 x float> @test_vfmlal_high_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c) #0 {
entry:
; CHECK-LABEL: test_vfmlal_high_u32:
; CHECK: fmlal2   v0.2s, v1.2h, v2.2h
  %vfmlal_high2.i = call <2 x float> @llvm.aarch64.neon.fmlal2.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> %c) #2
  ret <2 x float> %vfmlal_high2.i
}

define <2 x float> @test_vfmlsl_high_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c) #0 {
entry:
; CHECK-LABEL: test_vfmlsl_high_u32:
; CHECK: fmlsl2   v0.2s, v1.2h, v2.2h
  %vfmlsl_high2.i = call <2 x float> @llvm.aarch64.neon.fmlsl2.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> %c) #2
  ret <2 x float> %vfmlsl_high2.i
}

define <4 x float> @test_vfmlalq_low_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c) #0 {
entry:
; CHECK-LABEL: test_vfmlalq_low_u32:
; CHECK: fmlal   v0.4s, v1.4h, v2.4h
  %vfmlalq_low4.i = call <4 x float> @llvm.aarch64.neon.fmlal.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> %c) #2
  ret <4 x float> %vfmlalq_low4.i
}

define <4 x float> @test_vfmlslq_low_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c) #0 {
entry:
; CHECK-LABEL: test_vfmlslq_low_u32:
; CHECK: fmlsl   v0.4s, v1.4h, v2.4h
  %vfmlslq_low4.i = call <4 x float> @llvm.aarch64.neon.fmlsl.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> %c) #2
  ret <4 x float> %vfmlslq_low4.i
}

define <4 x float> @test_vfmlalq_high_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c) #0 {
entry:
; CHECK-LABEL: test_vfmlalq_high_u32:
; CHECK: fmlal2   v0.4s, v1.4h, v2.4h
  %vfmlalq_high4.i = call <4 x float> @llvm.aarch64.neon.fmlal2.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> %c) #2
  ret <4 x float> %vfmlalq_high4.i
}

define <4 x float> @test_vfmlslq_high_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c) #0 {
entry:
; CHECK-LABEL: test_vfmlslq_high_u32:
; CHECK: fmlsl2   v0.4s, v1.4h, v2.4h
  %vfmlslq_high4.i = call <4 x float> @llvm.aarch64.neon.fmlsl2.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> %c) #2
  ret <4 x float> %vfmlslq_high4.i
}
