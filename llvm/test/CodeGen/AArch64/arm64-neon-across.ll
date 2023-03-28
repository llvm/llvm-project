; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon | FileCheck %s
; RUN: llc < %s -global-isel=1 -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon | FileCheck %s

declare float @llvm.aarch64.neon.fminnmv.f32.v4f32(<4 x float>)

declare float @llvm.aarch64.neon.fmaxnmv.f32.v4f32(<4 x float>)

declare float @llvm.aarch64.neon.fminv.f32.v4f32(<4 x float>)

declare float @llvm.aarch64.neon.fmaxv.f32.v4f32(<4 x float>)

declare i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.saddv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.saddv.i32.v16i8(<16 x i8>)

declare i32 @llvm.aarch64.neon.saddv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.saddv.i32.v8i8(<8 x i8>)

declare i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.uminv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.uminv.i32.v16i8(<16 x i8>)

declare i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.sminv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.sminv.i32.v16i8(<16 x i8>)

declare i32 @llvm.aarch64.neon.uminv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.uminv.i32.v8i8(<8 x i8>)

declare i32 @llvm.aarch64.neon.sminv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.sminv.i32.v8i8(<8 x i8>)

declare i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.umaxv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.umaxv.i32.v16i8(<16 x i8>)

declare i32 @llvm.aarch64.neon.smaxv.i32.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.smaxv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.smaxv.i32.v16i8(<16 x i8>)

declare i32 @llvm.aarch64.neon.umaxv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.umaxv.i32.v8i8(<8 x i8>)

declare i32 @llvm.aarch64.neon.smaxv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.smaxv.i32.v8i8(<8 x i8>)

declare i64 @llvm.aarch64.neon.uaddlv.i64.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.uaddlv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.uaddlv.i32.v16i8(<16 x i8>)

declare i64 @llvm.aarch64.neon.saddlv.i64.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.saddlv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.saddlv.i32.v16i8(<16 x i8>)

declare i32 @llvm.aarch64.neon.uaddlv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.uaddlv.i32.v8i8(<8 x i8>)

declare i32 @llvm.aarch64.neon.saddlv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.saddlv.i32.v8i8(<8 x i8>)

define i16 @test_vaddlv_s8(<8 x i8> %a) {
; CHECK: test_vaddlv_s8:
; CHECK: saddlv h{{[0-9]+}}, {{v[0-9]+}}.8b
entry:
  %saddlvv.i = tail call i32 @llvm.aarch64.neon.saddlv.i32.v8i8(<8 x i8> %a)
  %0 = trunc i32 %saddlvv.i to i16
  ret i16 %0
}

define i32 @test_vaddlv_s16(<4 x i16> %a) {
; CHECK: test_vaddlv_s16:
; CHECK: saddlv s{{[0-9]+}}, {{v[0-9]+}}.4h
entry:
  %saddlvv.i = tail call i32 @llvm.aarch64.neon.saddlv.i32.v4i16(<4 x i16> %a)
  ret i32 %saddlvv.i
}

define i16 @test_vaddlv_u8(<8 x i8> %a) {
; CHECK: test_vaddlv_u8:
; CHECK: uaddlv h{{[0-9]+}}, {{v[0-9]+}}.8b
entry:
  %uaddlvv.i = tail call i32 @llvm.aarch64.neon.uaddlv.i32.v8i8(<8 x i8> %a)
  %0 = trunc i32 %uaddlvv.i to i16
  ret i16 %0
}

define i32 @test_vaddlv_u16(<4 x i16> %a) {
; CHECK: test_vaddlv_u16:
; CHECK: uaddlv s{{[0-9]+}}, {{v[0-9]+}}.4h
entry:
  %uaddlvv.i = tail call i32 @llvm.aarch64.neon.uaddlv.i32.v4i16(<4 x i16> %a)
  ret i32 %uaddlvv.i
}

define i16 @test_vaddlvq_s8(<16 x i8> %a) {
; CHECK: test_vaddlvq_s8:
; CHECK: saddlv h{{[0-9]+}}, {{v[0-9]+}}.16b
entry:
  %saddlvv.i = tail call i32 @llvm.aarch64.neon.saddlv.i32.v16i8(<16 x i8> %a)
  %0 = trunc i32 %saddlvv.i to i16
  ret i16 %0
}

define i32 @test_vaddlvq_s16(<8 x i16> %a) {
; CHECK: test_vaddlvq_s16:
; CHECK: saddlv s{{[0-9]+}}, {{v[0-9]+}}.8h
entry:
  %saddlvv.i = tail call i32 @llvm.aarch64.neon.saddlv.i32.v8i16(<8 x i16> %a)
  ret i32 %saddlvv.i
}

define i64 @test_vaddlvq_s32(<4 x i32> %a) {
; CHECK: test_vaddlvq_s32:
; CHECK: saddlv d{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %saddlvv.i = tail call i64 @llvm.aarch64.neon.saddlv.i64.v4i32(<4 x i32> %a)
  ret i64 %saddlvv.i
}

define i16 @test_vaddlvq_u8(<16 x i8> %a) {
; CHECK: test_vaddlvq_u8:
; CHECK: uaddlv h{{[0-9]+}}, {{v[0-9]+}}.16b
entry:
  %uaddlvv.i = tail call i32 @llvm.aarch64.neon.uaddlv.i32.v16i8(<16 x i8> %a)
  %0 = trunc i32 %uaddlvv.i to i16
  ret i16 %0
}

define i32 @test_vaddlvq_u16(<8 x i16> %a) {
; CHECK: test_vaddlvq_u16:
; CHECK: uaddlv s{{[0-9]+}}, {{v[0-9]+}}.8h
entry:
  %uaddlvv.i = tail call i32 @llvm.aarch64.neon.uaddlv.i32.v8i16(<8 x i16> %a)
  ret i32 %uaddlvv.i
}

define i64 @test_vaddlvq_u32(<4 x i32> %a) {
; CHECK: test_vaddlvq_u32:
; CHECK: uaddlv d{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %uaddlvv.i = tail call i64 @llvm.aarch64.neon.uaddlv.i64.v4i32(<4 x i32> %a)
  ret i64 %uaddlvv.i
}

define i8 @test_vmaxv_s8(<8 x i8> %a) {
; CHECK: test_vmaxv_s8:
; CHECK: smaxv b{{[0-9]+}}, {{v[0-9]+}}.8b
entry:
  %smaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v8i8(<8 x i8> %a)
  %0 = trunc i32 %smaxv.i to i8
  ret i8 %0
}

define i16 @test_vmaxv_s16(<4 x i16> %a) {
; CHECK: test_vmaxv_s16:
; CHECK: smaxv h{{[0-9]+}}, {{v[0-9]+}}.4h
entry:
  %smaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v4i16(<4 x i16> %a)
  %0 = trunc i32 %smaxv.i to i16
  ret i16 %0
}

define i8 @test_vmaxv_u8(<8 x i8> %a) {
; CHECK: test_vmaxv_u8:
; CHECK: umaxv b{{[0-9]+}}, {{v[0-9]+}}.8b
entry:
  %umaxv.i = tail call i32 @llvm.aarch64.neon.umaxv.i32.v8i8(<8 x i8> %a)
  %0 = trunc i32 %umaxv.i to i8
  ret i8 %0
}

define i16 @test_vmaxv_u16(<4 x i16> %a) {
; CHECK: test_vmaxv_u16:
; CHECK: umaxv h{{[0-9]+}}, {{v[0-9]+}}.4h
entry:
  %umaxv.i = tail call i32 @llvm.aarch64.neon.umaxv.i32.v4i16(<4 x i16> %a)
  %0 = trunc i32 %umaxv.i to i16
  ret i16 %0
}

define i8 @test_vmaxvq_s8(<16 x i8> %a) {
; CHECK: test_vmaxvq_s8:
; CHECK: smaxv b{{[0-9]+}}, {{v[0-9]+}}.16b
entry:
  %smaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v16i8(<16 x i8> %a)
  %0 = trunc i32 %smaxv.i to i8
  ret i8 %0
}

define i16 @test_vmaxvq_s16(<8 x i16> %a) {
; CHECK: test_vmaxvq_s16:
; CHECK: smaxv h{{[0-9]+}}, {{v[0-9]+}}.8h
entry:
  %smaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v8i16(<8 x i16> %a)
  %0 = trunc i32 %smaxv.i to i16
  ret i16 %0
}

define i32 @test_vmaxvq_s32(<4 x i32> %a) {
; CHECK: test_vmaxvq_s32:
; CHECK: smaxv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %smaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v4i32(<4 x i32> %a)
  ret i32 %smaxv.i
}

define i8 @test_vmaxvq_u8(<16 x i8> %a) {
; CHECK: test_vmaxvq_u8:
; CHECK: umaxv b{{[0-9]+}}, {{v[0-9]+}}.16b
entry:
  %umaxv.i = tail call i32 @llvm.aarch64.neon.umaxv.i32.v16i8(<16 x i8> %a)
  %0 = trunc i32 %umaxv.i to i8
  ret i8 %0
}

define i16 @test_vmaxvq_u16(<8 x i16> %a) {
; CHECK: test_vmaxvq_u16:
; CHECK: umaxv h{{[0-9]+}}, {{v[0-9]+}}.8h
entry:
  %umaxv.i = tail call i32 @llvm.aarch64.neon.umaxv.i32.v8i16(<8 x i16> %a)
  %0 = trunc i32 %umaxv.i to i16
  ret i16 %0
}

define i32 @test_vmaxvq_u32(<4 x i32> %a) {
; CHECK: test_vmaxvq_u32:
; CHECK: umaxv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %umaxv.i = tail call i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32> %a)
  ret i32 %umaxv.i
}

define i8 @test_vminv_s8(<8 x i8> %a) {
; CHECK: test_vminv_s8:
; CHECK: sminv b{{[0-9]+}}, {{v[0-9]+}}.8b
entry:
  %sminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v8i8(<8 x i8> %a)
  %0 = trunc i32 %sminv.i to i8
  ret i8 %0
}

define i16 @test_vminv_s16(<4 x i16> %a) {
; CHECK: test_vminv_s16:
; CHECK: sminv h{{[0-9]+}}, {{v[0-9]+}}.4h
entry:
  %sminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v4i16(<4 x i16> %a)
  %0 = trunc i32 %sminv.i to i16
  ret i16 %0
}

define i8 @test_vminv_u8(<8 x i8> %a) {
; CHECK: test_vminv_u8:
; CHECK: uminv b{{[0-9]+}}, {{v[0-9]+}}.8b
entry:
  %uminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v8i8(<8 x i8> %a)
  %0 = trunc i32 %uminv.i to i8
  ret i8 %0
}

define i16 @test_vminv_u16(<4 x i16> %a) {
; CHECK: test_vminv_u16:
; CHECK: uminv h{{[0-9]+}}, {{v[0-9]+}}.4h
entry:
  %uminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v4i16(<4 x i16> %a)
  %0 = trunc i32 %uminv.i to i16
  ret i16 %0
}

define i8 @test_vminvq_s8(<16 x i8> %a) {
; CHECK: test_vminvq_s8:
; CHECK: sminv b{{[0-9]+}}, {{v[0-9]+}}.16b
entry:
  %sminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v16i8(<16 x i8> %a)
  %0 = trunc i32 %sminv.i to i8
  ret i8 %0
}

define i16 @test_vminvq_s16(<8 x i16> %a) {
; CHECK: test_vminvq_s16:
; CHECK: sminv h{{[0-9]+}}, {{v[0-9]+}}.8h
entry:
  %sminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v8i16(<8 x i16> %a)
  %0 = trunc i32 %sminv.i to i16
  ret i16 %0
}

define i32 @test_vminvq_s32(<4 x i32> %a) {
; CHECK: test_vminvq_s32:
; CHECK: sminv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %sminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32> %a)
  ret i32 %sminv.i
}

define i8 @test_vminvq_u8(<16 x i8> %a) {
; CHECK: test_vminvq_u8:
; CHECK: uminv b{{[0-9]+}}, {{v[0-9]+}}.16b
entry:
  %uminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v16i8(<16 x i8> %a)
  %0 = trunc i32 %uminv.i to i8
  ret i8 %0
}

define i16 @test_vminvq_u16(<8 x i16> %a) {
; CHECK: test_vminvq_u16:
; CHECK: uminv h{{[0-9]+}}, {{v[0-9]+}}.8h
entry:
  %uminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v8i16(<8 x i16> %a)
  %0 = trunc i32 %uminv.i to i16
  ret i16 %0
}

define i32 @test_vminvq_u32(<4 x i32> %a) {
; CHECK: test_vminvq_u32:
; CHECK: uminv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %uminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32> %a)
  ret i32 %uminv.i
}

define i8 @test_vaddv_s8(<8 x i8> %a) {
; CHECK: test_vaddv_s8:
; CHECK: addv b{{[0-9]+}}, {{v[0-9]+}}.8b
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v8i8(<8 x i8> %a)
  %0 = trunc i32 %vaddv.i to i8
  ret i8 %0
}

define i16 @test_vaddv_s16(<4 x i16> %a) {
; CHECK: test_vaddv_s16:
; CHECK: addv h{{[0-9]+}}, {{v[0-9]+}}.4h
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v4i16(<4 x i16> %a)
  %0 = trunc i32 %vaddv.i to i16
  ret i16 %0
}

define i8 @test_vaddv_u8(<8 x i8> %a) {
; CHECK: test_vaddv_u8:
; CHECK: addv b{{[0-9]+}}, {{v[0-9]+}}.8b
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v8i8(<8 x i8> %a)
  %0 = trunc i32 %vaddv.i to i8
  ret i8 %0
}

define i16 @test_vaddv_u16(<4 x i16> %a) {
; CHECK: test_vaddv_u16:
; CHECK: addv h{{[0-9]+}}, {{v[0-9]+}}.4h
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v4i16(<4 x i16> %a)
  %0 = trunc i32 %vaddv.i to i16
  ret i16 %0
}

define i8 @test_vaddvq_s8(<16 x i8> %a) {
; CHECK: test_vaddvq_s8:
; CHECK: addv b{{[0-9]+}}, {{v[0-9]+}}.16b
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v16i8(<16 x i8> %a)
  %0 = trunc i32 %vaddv.i to i8
  ret i8 %0
}

define i16 @test_vaddvq_s16(<8 x i16> %a) {
; CHECK: test_vaddvq_s16:
; CHECK: addv h{{[0-9]+}}, {{v[0-9]+}}.8h
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v8i16(<8 x i16> %a)
  %0 = trunc i32 %vaddv.i to i16
  ret i16 %0
}

define i32 @test_vaddvq_s32(<4 x i32> %a) {
; CHECK: test_vaddvq_s32:
; CHECK: addv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32> %a)
  ret i32 %vaddv.i
}

define i8 @test_vaddvq_u8(<16 x i8> %a) {
; CHECK: test_vaddvq_u8:
; CHECK: addv b{{[0-9]+}}, {{v[0-9]+}}.16b
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v16i8(<16 x i8> %a)
  %0 = trunc i32 %vaddv.i to i8
  ret i8 %0
}

define i16 @test_vaddvq_u16(<8 x i16> %a) {
; CHECK: test_vaddvq_u16:
; CHECK: addv h{{[0-9]+}}, {{v[0-9]+}}.8h
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v8i16(<8 x i16> %a)
  %0 = trunc i32 %vaddv.i to i16
  ret i16 %0
}

define i32 @test_vaddvq_u32(<4 x i32> %a) {
; CHECK: test_vaddvq_u32:
; CHECK: addv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32> %a)
  ret i32 %vaddv.i
}

define float @test_vmaxvq_f32(<4 x float> %a) {
; CHECK: test_vmaxvq_f32:
; CHECK: fmaxv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %0 = call float @llvm.aarch64.neon.fmaxv.f32.v4f32(<4 x float> %a)
  ret float %0
}

define float @test_vminvq_f32(<4 x float> %a) {
; CHECK: test_vminvq_f32:
; CHECK: fminv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %0 = call float @llvm.aarch64.neon.fminv.f32.v4f32(<4 x float> %a)
  ret float %0
}

define float @test_vmaxnmvq_f32(<4 x float> %a) {
; CHECK: test_vmaxnmvq_f32:
; CHECK: fmaxnmv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %0 = call float @llvm.aarch64.neon.fmaxnmv.f32.v4f32(<4 x float> %a)
  ret float %0
}

define float @test_vminnmvq_f32(<4 x float> %a) {
; CHECK: test_vminnmvq_f32:
; CHECK: fminnmv s{{[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %0 = call float @llvm.aarch64.neon.fminnmv.f32.v4f32(<4 x float> %a)
  ret float %0
}

