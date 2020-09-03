; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-arm-none-eabi -mattr=+neon -mattr=+bf16 | FileCheck %s

declare bfloat @llvm.aarch64.neon.bfcvt(float)
declare <8 x bfloat> @llvm.aarch64.neon.bfcvtn(<4 x float>)
declare <8 x bfloat> @llvm.aarch64.neon.bfcvtn2(<8 x bfloat>, <4 x float>)

; CHECK-LABEL: test_vcvth_bf16_f32
; CHECK:      bfcvt h0, s0
; CHECK-NEXT: ret
define bfloat @test_vcvth_bf16_f32(float %a) {
entry:
  %vcvth_bf16_f32 = call bfloat @llvm.aarch64.neon.bfcvt(float %a)
  ret bfloat %vcvth_bf16_f32
}

; CHECK-LABEL: test_vcvtq_low_bf16_f32
; CHECK:      bfcvtn v0.4h, v0.4s
; CHECK-NEXT: ret
define <8 x bfloat> @test_vcvtq_low_bf16_f32(<4 x float> %a) {
entry:
  %cvt = call <8 x bfloat> @llvm.aarch64.neon.bfcvtn(<4 x float> %a)
  ret <8 x bfloat> %cvt
}

; CHECK-LABEL: test_vcvtq_high_bf16_f32
; CHECK:      bfcvtn2 v1.8h, v0.4s
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
define <8 x bfloat> @test_vcvtq_high_bf16_f32(<4 x float> %a, <8 x bfloat> %inactive) {
entry:
  %cvt = call <8 x bfloat> @llvm.aarch64.neon.bfcvtn2(<8 x bfloat> %inactive, <4 x float> %a)
  ret <8 x bfloat> %cvt
}

