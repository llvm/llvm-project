; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s

declare <2 x double> @llvm.aarch64.neon.vmulx.v2f64(<2 x double>, <2 x double>)

declare <4 x float> @llvm.aarch64.neon.vmulx.v4f32(<4 x float>, <4 x float>)

declare <2 x float> @llvm.aarch64.neon.vmulx.v2f32(<2 x float>, <2 x float>)

declare <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32>, <2 x i32>)

declare <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16>, <8 x i16>)

declare <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16>, <4 x i16>)

declare <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32>, <2 x i32>)

declare <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16>, <8 x i16>)

declare <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16>, <4 x i16>)

declare <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32>, <2 x i32>)

declare <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16>, <4 x i16>)

declare <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64>, <2 x i64>)

declare <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64>, <2 x i64>)

declare <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32>, <2 x i32>)

declare <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16>, <4 x i16>)

declare <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32>, <2 x i32>)

declare <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16>, <4 x i16>)

define <4 x i16> @test_vmla_lane_s16(<4 x i16> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmla_lane_s16:
; CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = mul <4 x i16> %shuffle, %b
  %add = add <4 x i16> %mul, %a
  ret <4 x i16> %add
}

define <8 x i16> @test_vmlaq_lane_s16(<8 x i16> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlaq_lane_s16:
; CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %mul = mul <8 x i16> %shuffle, %b
  %add = add <8 x i16> %mul, %a
  ret <8 x i16> %add
}

define <2 x i32> @test_vmla_lane_s32(<2 x i32> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmla_lane_s32:
; CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %mul = mul <2 x i32> %shuffle, %b
  %add = add <2 x i32> %mul, %a
  ret <2 x i32> %add
}

define <4 x i32> @test_vmlaq_lane_s32(<4 x i32> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlaq_lane_s32:
; CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mul = mul <4 x i32> %shuffle, %b
  %add = add <4 x i32> %mul, %a
  ret <4 x i32> %add
}

define <4 x i16> @test_vmla_laneq_s16(<4 x i16> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmla_laneq_s16:
; CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mul = mul <4 x i16> %shuffle, %b
  %add = add <4 x i16> %mul, %a
  ret <4 x i16> %add
}

define <8 x i16> @test_vmlaq_laneq_s16(<8 x i16> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlaq_laneq_s16:
; CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %mul = mul <8 x i16> %shuffle, %b
  %add = add <8 x i16> %mul, %a
  ret <8 x i16> %add
}

define <2 x i32> @test_vmla_laneq_s32(<2 x i32> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmla_laneq_s32:
; CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %mul = mul <2 x i32> %shuffle, %b
  %add = add <2 x i32> %mul, %a
  ret <2 x i32> %add
}

define <4 x i32> @test_vmlaq_laneq_s32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlaq_laneq_s32:
; CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = mul <4 x i32> %shuffle, %b
  %add = add <4 x i32> %mul, %a
  ret <4 x i32> %add
}

define <4 x i16> @test_vmls_lane_s16(<4 x i16> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmls_lane_s16:
; CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = mul <4 x i16> %shuffle, %b
  %sub = sub <4 x i16> %a, %mul
  ret <4 x i16> %sub
}

define <8 x i16> @test_vmlsq_lane_s16(<8 x i16> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsq_lane_s16:
; CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %mul = mul <8 x i16> %shuffle, %b
  %sub = sub <8 x i16> %a, %mul
  ret <8 x i16> %sub
}

define <2 x i32> @test_vmls_lane_s32(<2 x i32> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmls_lane_s32:
; CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %mul = mul <2 x i32> %shuffle, %b
  %sub = sub <2 x i32> %a, %mul
  ret <2 x i32> %sub
}

define <4 x i32> @test_vmlsq_lane_s32(<4 x i32> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsq_lane_s32:
; CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mul = mul <4 x i32> %shuffle, %b
  %sub = sub <4 x i32> %a, %mul
  ret <4 x i32> %sub
}

define <4 x i16> @test_vmls_laneq_s16(<4 x i16> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmls_laneq_s16:
; CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mul = mul <4 x i16> %shuffle, %b
  %sub = sub <4 x i16> %a, %mul
  ret <4 x i16> %sub
}

define <8 x i16> @test_vmlsq_laneq_s16(<8 x i16> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsq_laneq_s16:
; CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %mul = mul <8 x i16> %shuffle, %b
  %sub = sub <8 x i16> %a, %mul
  ret <8 x i16> %sub
}

define <2 x i32> @test_vmls_laneq_s32(<2 x i32> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmls_laneq_s32:
; CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %mul = mul <2 x i32> %shuffle, %b
  %sub = sub <2 x i32> %a, %mul
  ret <2 x i32> %sub
}

define <4 x i32> @test_vmlsq_laneq_s32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsq_laneq_s32:
; CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = mul <4 x i32> %shuffle, %b
  %sub = sub <4 x i32> %a, %mul
  ret <4 x i32> %sub
}

define <4 x i16> @test_vmul_lane_s16(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmul_lane_s16:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = mul <4 x i16> %shuffle, %a
  ret <4 x i16> %mul
}

define <8 x i16> @test_vmulq_lane_s16(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmulq_lane_s16:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %mul = mul <8 x i16> %shuffle, %a
  ret <8 x i16> %mul
}

define <2 x i32> @test_vmul_lane_s32(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmul_lane_s32:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %mul = mul <2 x i32> %shuffle, %a
  ret <2 x i32> %mul
}

define <4 x i32> @test_vmulq_lane_s32(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmulq_lane_s32:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mul = mul <4 x i32> %shuffle, %a
  ret <4 x i32> %mul
}

define <4 x i16> @test_vmul_lane_u16(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmul_lane_u16:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = mul <4 x i16> %shuffle, %a
  ret <4 x i16> %mul
}

define <8 x i16> @test_vmulq_lane_u16(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmulq_lane_u16:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %mul = mul <8 x i16> %shuffle, %a
  ret <8 x i16> %mul
}

define <2 x i32> @test_vmul_lane_u32(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmul_lane_u32:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %mul = mul <2 x i32> %shuffle, %a
  ret <2 x i32> %mul
}

define <4 x i32> @test_vmulq_lane_u32(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmulq_lane_u32:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mul = mul <4 x i32> %shuffle, %a
  ret <4 x i32> %mul
}

define <4 x i16> @test_vmul_laneq_s16(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmul_laneq_s16:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mul = mul <4 x i16> %shuffle, %a
  ret <4 x i16> %mul
}

define <8 x i16> @test_vmulq_laneq_s16(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmulq_laneq_s16:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %mul = mul <8 x i16> %shuffle, %a
  ret <8 x i16> %mul
}

define <2 x i32> @test_vmul_laneq_s32(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmul_laneq_s32:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %mul = mul <2 x i32> %shuffle, %a
  ret <2 x i32> %mul
}

define <4 x i32> @test_vmulq_laneq_s32(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmulq_laneq_s32:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = mul <4 x i32> %shuffle, %a
  ret <4 x i32> %mul
}

define <4 x i16> @test_vmul_laneq_u16(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmul_laneq_u16:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mul = mul <4 x i16> %shuffle, %a
  ret <4 x i16> %mul
}

define <8 x i16> @test_vmulq_laneq_u16(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmulq_laneq_u16:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %mul = mul <8 x i16> %shuffle, %a
  ret <8 x i16> %mul
}

define <2 x i32> @test_vmul_laneq_u32(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmul_laneq_u32:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %mul = mul <2 x i32> %shuffle, %a
  ret <2 x i32> %mul
}

define <4 x i32> @test_vmulq_laneq_u32(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmulq_laneq_u32:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = mul <4 x i32> %shuffle, %a
  ret <4 x i32> %mul
}

define <2 x float> @test_vfma_lane_f32(<2 x float> %a, <2 x float> %b, <2 x float> %v) {
; CHECK: test_vfma_lane_f32:
; CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %lane = shufflevector <2 x float> %v, <2 x float> undef, <2 x i32> <i32 1, i32 1>
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %lane, <2 x float> %b, <2 x float> %a)
  ret <2 x float> %0
}

declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)

define <4 x float> @test_vfmaq_lane_f32(<4 x float> %a, <4 x float> %b, <2 x float> %v) {
; CHECK: test_vfmaq_lane_f32:
; CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %lane = shufflevector <2 x float> %v, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %lane, <4 x float> %b, <4 x float> %a)
  ret <4 x float> %0
}

declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)

define <2 x float> @test_vfma_laneq_f32(<2 x float> %a, <2 x float> %b, <4 x float> %v) {
; CHECK: test_vfma_laneq_f32:
; CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %lane = shufflevector <4 x float> %v, <4 x float> undef, <2 x i32> <i32 3, i32 3>
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %lane, <2 x float> %b, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmaq_laneq_f32(<4 x float> %a, <4 x float> %b, <4 x float> %v) {
; CHECK: test_vfmaq_laneq_f32:
; CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %lane = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %lane, <4 x float> %b, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x float> @test_vfms_lane_f32(<2 x float> %a, <2 x float> %b, <2 x float> %v) {
; CHECK: test_vfms_lane_f32:
; CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %sub = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %v
  %lane = shufflevector <2 x float> %sub, <2 x float> undef, <2 x i32> <i32 1, i32 1>
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %lane, <2 x float> %b, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmsq_lane_f32(<4 x float> %a, <4 x float> %b, <2 x float> %v) {
; CHECK: test_vfmsq_lane_f32:
; CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %sub = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %v
  %lane = shufflevector <2 x float> %sub, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %lane, <4 x float> %b, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x float> @test_vfms_laneq_f32(<2 x float> %a, <2 x float> %b, <4 x float> %v) {
; CHECK: test_vfms_laneq_f32:
; CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %sub = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %v
  %lane = shufflevector <4 x float> %sub, <4 x float> undef, <2 x i32> <i32 3, i32 3>
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %lane, <2 x float> %b, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmsq_laneq_f32(<4 x float> %a, <4 x float> %b, <4 x float> %v) {
; CHECK: test_vfmsq_laneq_f32:
; CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %sub = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %v
  %lane = shufflevector <4 x float> %sub, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %lane, <4 x float> %b, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x double> @test_vfmaq_lane_f64(<2 x double> %a, <2 x double> %b, <1 x double> %v) {
; CHECK: test_vfmaq_lane_f64:
; CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %lane = shufflevector <1 x double> %v, <1 x double> undef, <2 x i32> zeroinitializer
  %0 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %lane, <2 x double> %b, <2 x double> %a)
  ret <2 x double> %0
}

declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)

define <2 x double> @test_vfmaq_laneq_f64(<2 x double> %a, <2 x double> %b, <2 x double> %v) {
; CHECK: test_vfmaq_laneq_f64:
; CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
entry:
  %lane = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  %0 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %lane, <2 x double> %b, <2 x double> %a)
  ret <2 x double> %0
}

define <2 x double> @test_vfmsq_lane_f64(<2 x double> %a, <2 x double> %b, <1 x double> %v) {
; CHECK: test_vfmsq_lane_f64:
; CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %sub = fsub <1 x double> <double -0.000000e+00>, %v
  %lane = shufflevector <1 x double> %sub, <1 x double> undef, <2 x i32> zeroinitializer
  %0 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %lane, <2 x double> %b, <2 x double> %a)
  ret <2 x double> %0
}

define <2 x double> @test_vfmsq_laneq_f64(<2 x double> %a, <2 x double> %b, <2 x double> %v) {
; CHECK: test_vfmsq_laneq_f64:
; CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
entry:
  %sub = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %v
  %lane = shufflevector <2 x double> %sub, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  %0 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %lane, <2 x double> %b, <2 x double> %a)
  ret <2 x double> %0
}

define <4 x i32> @test_vmlal_lane_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlal_lane_s16:
; CHECK: mlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_lane_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlal_lane_s32:
; CHECK: mlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_laneq_s16(<4 x i32> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlal_laneq_s16:
; CHECK: mlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_laneq_s32(<2 x i64> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlal_laneq_s32:
; CHECK: mlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_high_lane_s16(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlal_high_lane_s16:
; CHECK: mlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_high_lane_s32(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlal_high_lane_s32:
; CHECK: mlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_high_laneq_s16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlal_high_laneq_s16:
; CHECK: mlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_high_laneq_s32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlal_high_laneq_s32:
; CHECK: mlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlsl_lane_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsl_lane_s16:
; CHECK: mlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_lane_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsl_lane_s32:
; CHECK: mlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_laneq_s16(<4 x i32> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsl_laneq_s16:
; CHECK: mlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_laneq_s32(<2 x i64> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsl_laneq_s32:
; CHECK: mlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_high_lane_s16(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsl_high_lane_s16:
; CHECK: mlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_high_lane_s32(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsl_high_lane_s32:
; CHECK: mlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_high_laneq_s16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsl_high_laneq_s16:
; CHECK: mlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_high_laneq_s32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsl_high_laneq_s32:
; CHECK: mlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlal_lane_u16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlal_lane_u16:
; CHECK: mlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_lane_u32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlal_lane_u32:
; CHECK: mlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_laneq_u16(<4 x i32> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlal_laneq_u16:
; CHECK: mlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_laneq_u32(<2 x i64> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlal_laneq_u32:
; CHECK: mlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_high_lane_u16(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlal_high_lane_u16:
; CHECK: mlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_high_lane_u32(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlal_high_lane_u32:
; CHECK: mlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_high_laneq_u16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlal_high_laneq_u16:
; CHECK: mlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_high_laneq_u32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlal_high_laneq_u32:
; CHECK: mlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlsl_lane_u16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsl_lane_u16:
; CHECK: mlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_lane_u32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsl_lane_u32:
; CHECK: mlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_laneq_u16(<4 x i32> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsl_laneq_u16:
; CHECK: mlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_laneq_u32(<2 x i64> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsl_laneq_u32:
; CHECK: mlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_high_lane_u16(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsl_high_lane_u16:
; CHECK: mlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_high_lane_u32(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsl_high_lane_u32:
; CHECK: mlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_high_laneq_u16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsl_high_laneq_u16:
; CHECK: mlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_high_laneq_u32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsl_high_laneq_u32:
; CHECK: mlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmull_lane_s16(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmull_lane_s16:
; CHECK: mull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_lane_s32(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmull_lane_s32:
; CHECK: mull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_lane_u16(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmull_lane_u16:
; CHECK: mull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_lane_u32(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmull_lane_u32:
; CHECK: mull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_high_lane_s16(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmull_high_lane_s16:
; CHECK: mull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_high_lane_s32(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmull_high_lane_s32:
; CHECK: mull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_high_lane_u16(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmull_high_lane_u16:
; CHECK: mull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_high_lane_u32(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmull_high_lane_u32:
; CHECK: mull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_laneq_s16(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmull_laneq_s16:
; CHECK: mull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_laneq_s32(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmull_laneq_s32:
; CHECK: mull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_laneq_u16(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmull_laneq_u16:
; CHECK: mull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_laneq_u32(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmull_laneq_u32:
; CHECK: mull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_high_laneq_s16(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmull_high_laneq_s16:
; CHECK: mull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_high_laneq_s32(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmull_high_laneq_s32:
; CHECK: mull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_high_laneq_u16(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmull_high_laneq_u16:
; CHECK: mull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_high_laneq_u32(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmull_high_laneq_u32:
; CHECK: mull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vqdmlal_lane_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vqdmlal_lane_s16:
; CHECK: qdmlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqdmlal2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %vqdmlal4.i = tail call <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal2.i)
  ret <4 x i32> %vqdmlal4.i
}

define <2 x i64> @test_vqdmlal_lane_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vqdmlal_lane_s32:
; CHECK: qdmlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vqdmlal2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %vqdmlal4.i = tail call <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal2.i)
  ret <2 x i64> %vqdmlal4.i
}

define <4 x i32> @test_vqdmlal_high_lane_s16(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vqdmlal_high_lane_s16:
; CHECK: qdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqdmlal2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %vqdmlal4.i = tail call <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal2.i)
  ret <4 x i32> %vqdmlal4.i
}

define <2 x i64> @test_vqdmlal_high_lane_s32(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vqdmlal_high_lane_s32:
; CHECK: qdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vqdmlal2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %vqdmlal4.i = tail call <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal2.i)
  ret <2 x i64> %vqdmlal4.i
}

define <4 x i32> @test_vqdmlsl_lane_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vqdmlsl_lane_s16:
; CHECK: qdmlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqdmlsl2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %vqdmlsl4.i = tail call <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl2.i)
  ret <4 x i32> %vqdmlsl4.i
}

define <2 x i64> @test_vqdmlsl_lane_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vqdmlsl_lane_s32:
; CHECK: qdmlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vqdmlsl2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %vqdmlsl4.i = tail call <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl2.i)
  ret <2 x i64> %vqdmlsl4.i
}

define <4 x i32> @test_vqdmlsl_high_lane_s16(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vqdmlsl_high_lane_s16:
; CHECK: qdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqdmlsl2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %vqdmlsl4.i = tail call <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl2.i)
  ret <4 x i32> %vqdmlsl4.i
}

define <2 x i64> @test_vqdmlsl_high_lane_s32(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vqdmlsl_high_lane_s32:
; CHECK: qdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vqdmlsl2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %vqdmlsl4.i = tail call <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl2.i)
  ret <2 x i64> %vqdmlsl4.i
}

define <4 x i32> @test_vqdmull_lane_s16(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqdmull_lane_s16:
; CHECK: qdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqdmull2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_lane_s32(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqdmull_lane_s32:
; CHECK: qdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vqdmull2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i32> @test_vqdmull_laneq_s16(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vqdmull_laneq_s16:
; CHECK: qdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqdmull2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_laneq_s32(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vqdmull_laneq_s32:
; CHECK: qdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vqdmull2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i32> @test_vqdmull_high_lane_s16(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqdmull_high_lane_s16:
; CHECK: qdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqdmull2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_high_lane_s32(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqdmull_high_lane_s32:
; CHECK: qdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vqdmull2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i32> @test_vqdmull_high_laneq_s16(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vqdmull_high_laneq_s16:
; CHECK: qdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %vqdmull2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_high_laneq_s32(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vqdmull_high_laneq_s32:
; CHECK: qdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> <i32 3, i32 3>
  %vqdmull2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i16> @test_vqdmulh_lane_s16(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqdmulh_lane_s16:
; CHECK: qdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqdmulh2.i = tail call <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i16> %vqdmulh2.i
}

define <8 x i16> @test_vqdmulhq_lane_s16(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqdmulhq_lane_s16:
; CHECK: qdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %vqdmulh2.i = tail call <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16> %a, <8 x i16> %shuffle)
  ret <8 x i16> %vqdmulh2.i
}

define <2 x i32> @test_vqdmulh_lane_s32(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqdmulh_lane_s32:
; CHECK: qdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vqdmulh2.i = tail call <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i32> %vqdmulh2.i
}

define <4 x i32> @test_vqdmulhq_lane_s32(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqdmulhq_lane_s32:
; CHECK: qdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %vqdmulh2.i = tail call <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32> %a, <4 x i32> %shuffle)
  ret <4 x i32> %vqdmulh2.i
}

define <4 x i16> @test_vqrdmulh_lane_s16(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqrdmulh_lane_s16:
; CHECK: qrdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vqrdmulh2.i = tail call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i16> %vqrdmulh2.i
}

define <8 x i16> @test_vqrdmulhq_lane_s16(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqrdmulhq_lane_s16:
; CHECK: qrdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %vqrdmulh2.i = tail call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %a, <8 x i16> %shuffle)
  ret <8 x i16> %vqrdmulh2.i
}

define <2 x i32> @test_vqrdmulh_lane_s32(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqrdmulh_lane_s32:
; CHECK: qrdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %vqrdmulh2.i = tail call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i32> %vqrdmulh2.i
}

define <4 x i32> @test_vqrdmulhq_lane_s32(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqrdmulhq_lane_s32:
; CHECK: qrdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %vqrdmulh2.i = tail call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %a, <4 x i32> %shuffle)
  ret <4 x i32> %vqrdmulh2.i
}

define <2 x float> @test_vmul_lane_f32(<2 x float> %a, <2 x float> %v) {
; CHECK: test_vmul_lane_f32:
; CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <2 x i32> <i32 1, i32 1>
  %mul = fmul <2 x float> %shuffle, %a
  ret <2 x float> %mul
}

define <1 x double> @test_vmul_lane_f64(<1 x double> %a, <1 x double> %v) {
; CHECK: test_vmul_lane_f64:
; CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
entry:
  %0 = bitcast <1 x double> %a to <8 x i8>
  %1 = bitcast <8 x i8> %0 to double
  %extract = extractelement <1 x double> %v, i32 0
  %2 = fmul double %1, %extract
  %3 = insertelement <1 x double> undef, double %2, i32 0
  ret <1 x double> %3
}

define <4 x float> @test_vmulq_lane_f32(<4 x float> %a, <2 x float> %v) {
; CHECK: test_vmulq_lane_f32:
; CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mul = fmul <4 x float> %shuffle, %a
  ret <4 x float> %mul
}

define <2 x double> @test_vmulq_lane_f64(<2 x double> %a, <1 x double> %v) {
; CHECK: test_vmulq_lane_f64:
; CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %shuffle = shufflevector <1 x double> %v, <1 x double> undef, <2 x i32> zeroinitializer
  %mul = fmul <2 x double> %shuffle, %a
  ret <2 x double> %mul
}

define <2 x float> @test_vmul_laneq_f32(<2 x float> %a, <4 x float> %v) {
; CHECK: test_vmul_laneq_f32:
; CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x float> %v, <4 x float> undef, <2 x i32> <i32 3, i32 3>
  %mul = fmul <2 x float> %shuffle, %a
  ret <2 x float> %mul
}

define <1 x double> @test_vmul_laneq_f64(<1 x double> %a, <2 x double> %v) {
; CHECK: test_vmul_laneq_f64:
; CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
entry:
  %0 = bitcast <1 x double> %a to <8 x i8>
  %1 = bitcast <8 x i8> %0 to double
  %extract = extractelement <2 x double> %v, i32 1
  %2 = fmul double %1, %extract
  %3 = insertelement <1 x double> undef, double %2, i32 0
  ret <1 x double> %3
}

define <4 x float> @test_vmulq_laneq_f32(<4 x float> %a, <4 x float> %v) {
; CHECK: test_vmulq_laneq_f32:
; CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = fmul <4 x float> %shuffle, %a
  ret <4 x float> %mul
}

define <2 x double> @test_vmulq_laneq_f64(<2 x double> %a, <2 x double> %v) {
; CHECK: test_vmulq_laneq_f64:
; CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
entry:
  %shuffle = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  %mul = fmul <2 x double> %shuffle, %a
  ret <2 x double> %mul
}

define <2 x float> @test_vmulx_lane_f32(<2 x float> %a, <2 x float> %v) {
; CHECK: test_vmulx_lane_f32:
; CHECK: mulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <2 x i32> <i32 1, i32 1>
  %vmulx2.i = tail call <2 x float> @llvm.aarch64.neon.vmulx.v2f32(<2 x float> %a, <2 x float> %shuffle)
  ret <2 x float> %vmulx2.i
}

define <4 x float> @test_vmulxq_lane_f32(<4 x float> %a, <2 x float> %v) {
; CHECK: test_vmulxq_lane_f32:
; CHECK: mulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %vmulx2.i = tail call <4 x float> @llvm.aarch64.neon.vmulx.v4f32(<4 x float> %a, <4 x float> %shuffle)
  ret <4 x float> %vmulx2.i
}

define <2 x double> @test_vmulxq_lane_f64(<2 x double> %a, <1 x double> %v) {
; CHECK: test_vmulxq_lane_f64:
; CHECK: mulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %shuffle = shufflevector <1 x double> %v, <1 x double> undef, <2 x i32> zeroinitializer
  %vmulx2.i = tail call <2 x double> @llvm.aarch64.neon.vmulx.v2f64(<2 x double> %a, <2 x double> %shuffle)
  ret <2 x double> %vmulx2.i
}

define <2 x float> @test_vmulx_laneq_f32(<2 x float> %a, <4 x float> %v) {
; CHECK: test_vmulx_laneq_f32:
; CHECK: mulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x float> %v, <4 x float> undef, <2 x i32> <i32 3, i32 3>
  %vmulx2.i = tail call <2 x float> @llvm.aarch64.neon.vmulx.v2f32(<2 x float> %a, <2 x float> %shuffle)
  ret <2 x float> %vmulx2.i
}

define <4 x float> @test_vmulxq_laneq_f32(<4 x float> %a, <4 x float> %v) {
; CHECK: test_vmulxq_laneq_f32:
; CHECK: mulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
entry:
  %shuffle = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %vmulx2.i = tail call <4 x float> @llvm.aarch64.neon.vmulx.v4f32(<4 x float> %a, <4 x float> %shuffle)
  ret <4 x float> %vmulx2.i
}

define <2 x double> @test_vmulxq_laneq_f64(<2 x double> %a, <2 x double> %v) {
; CHECK: test_vmulxq_laneq_f64:
; CHECK: mulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
entry:
  %shuffle = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  %vmulx2.i = tail call <2 x double> @llvm.aarch64.neon.vmulx.v2f64(<2 x double> %a, <2 x double> %shuffle)
  ret <2 x double> %vmulx2.i
}

define <4 x i16> @test_vmla_lane_s16_0(<4 x i16> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmla_lane_s16_0:
; CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i16> %shuffle, %b
  %add = add <4 x i16> %mul, %a
  ret <4 x i16> %add
}

define <8 x i16> @test_vmlaq_lane_s16_0(<8 x i16> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlaq_lane_s16_0:
; CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> zeroinitializer
  %mul = mul <8 x i16> %shuffle, %b
  %add = add <8 x i16> %mul, %a
  ret <8 x i16> %add
}

define <2 x i32> @test_vmla_lane_s32_0(<2 x i32> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmla_lane_s32_0:
; CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %b
  %add = add <2 x i32> %mul, %a
  ret <2 x i32> %add
}

define <4 x i32> @test_vmlaq_lane_s32_0(<4 x i32> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlaq_lane_s32_0:
; CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i32> %shuffle, %b
  %add = add <4 x i32> %mul, %a
  ret <4 x i32> %add
}

define <4 x i16> @test_vmla_laneq_s16_0(<4 x i16> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmla_laneq_s16_0:
; CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i16> %shuffle, %b
  %add = add <4 x i16> %mul, %a
  ret <4 x i16> %add
}

define <8 x i16> @test_vmlaq_laneq_s16_0(<8 x i16> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlaq_laneq_s16_0:
; CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> zeroinitializer
  %mul = mul <8 x i16> %shuffle, %b
  %add = add <8 x i16> %mul, %a
  ret <8 x i16> %add
}

define <2 x i32> @test_vmla_laneq_s32_0(<2 x i32> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmla_laneq_s32_0:
; CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %b
  %add = add <2 x i32> %mul, %a
  ret <2 x i32> %add
}

define <4 x i32> @test_vmlaq_laneq_s32_0(<4 x i32> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlaq_laneq_s32_0:
; CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i32> %shuffle, %b
  %add = add <4 x i32> %mul, %a
  ret <4 x i32> %add
}

define <4 x i16> @test_vmls_lane_s16_0(<4 x i16> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmls_lane_s16_0:
; CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i16> %shuffle, %b
  %sub = sub <4 x i16> %a, %mul
  ret <4 x i16> %sub
}

define <8 x i16> @test_vmlsq_lane_s16_0(<8 x i16> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsq_lane_s16_0:
; CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> zeroinitializer
  %mul = mul <8 x i16> %shuffle, %b
  %sub = sub <8 x i16> %a, %mul
  ret <8 x i16> %sub
}

define <2 x i32> @test_vmls_lane_s32_0(<2 x i32> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmls_lane_s32_0:
; CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %b
  %sub = sub <2 x i32> %a, %mul
  ret <2 x i32> %sub
}

define <4 x i32> @test_vmlsq_lane_s32_0(<4 x i32> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsq_lane_s32_0:
; CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i32> %shuffle, %b
  %sub = sub <4 x i32> %a, %mul
  ret <4 x i32> %sub
}

define <4 x i16> @test_vmls_laneq_s16_0(<4 x i16> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmls_laneq_s16_0:
; CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i16> %shuffle, %b
  %sub = sub <4 x i16> %a, %mul
  ret <4 x i16> %sub
}

define <8 x i16> @test_vmlsq_laneq_s16_0(<8 x i16> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsq_laneq_s16_0:
; CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> zeroinitializer
  %mul = mul <8 x i16> %shuffle, %b
  %sub = sub <8 x i16> %a, %mul
  ret <8 x i16> %sub
}

define <2 x i32> @test_vmls_laneq_s32_0(<2 x i32> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmls_laneq_s32_0:
; CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %b
  %sub = sub <2 x i32> %a, %mul
  ret <2 x i32> %sub
}

define <4 x i32> @test_vmlsq_laneq_s32_0(<4 x i32> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsq_laneq_s32_0:
; CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i32> %shuffle, %b
  %sub = sub <4 x i32> %a, %mul
  ret <4 x i32> %sub
}

define <4 x i16> @test_vmul_lane_s16_0(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmul_lane_s16_0:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i16> %shuffle, %a
  ret <4 x i16> %mul
}

define <8 x i16> @test_vmulq_lane_s16_0(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmulq_lane_s16_0:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> zeroinitializer
  %mul = mul <8 x i16> %shuffle, %a
  ret <8 x i16> %mul
}

define <2 x i32> @test_vmul_lane_s32_0(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmul_lane_s32_0:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %a
  ret <2 x i32> %mul
}

define <4 x i32> @test_vmulq_lane_s32_0(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmulq_lane_s32_0:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i32> %shuffle, %a
  ret <4 x i32> %mul
}

define <4 x i16> @test_vmul_lane_u16_0(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmul_lane_u16_0:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i16> %shuffle, %a
  ret <4 x i16> %mul
}

define <8 x i16> @test_vmulq_lane_u16_0(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmulq_lane_u16_0:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> zeroinitializer
  %mul = mul <8 x i16> %shuffle, %a
  ret <8 x i16> %mul
}

define <2 x i32> @test_vmul_lane_u32_0(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmul_lane_u32_0:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %a
  ret <2 x i32> %mul
}

define <4 x i32> @test_vmulq_lane_u32_0(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmulq_lane_u32_0:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i32> %shuffle, %a
  ret <4 x i32> %mul
}

define <4 x i16> @test_vmul_laneq_s16_0(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmul_laneq_s16_0:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i16> %shuffle, %a
  ret <4 x i16> %mul
}

define <8 x i16> @test_vmulq_laneq_s16_0(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmulq_laneq_s16_0:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> zeroinitializer
  %mul = mul <8 x i16> %shuffle, %a
  ret <8 x i16> %mul
}

define <2 x i32> @test_vmul_laneq_s32_0(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmul_laneq_s32_0:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %a
  ret <2 x i32> %mul
}

define <4 x i32> @test_vmulq_laneq_s32_0(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmulq_laneq_s32_0:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i32> %shuffle, %a
  ret <4 x i32> %mul
}

define <4 x i16> @test_vmul_laneq_u16_0(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmul_laneq_u16_0:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i16> %shuffle, %a
  ret <4 x i16> %mul
}

define <8 x i16> @test_vmulq_laneq_u16_0(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmulq_laneq_u16_0:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> zeroinitializer
  %mul = mul <8 x i16> %shuffle, %a
  ret <8 x i16> %mul
}

define <2 x i32> @test_vmul_laneq_u32_0(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmul_laneq_u32_0:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %a
  ret <2 x i32> %mul
}

define <4 x i32> @test_vmulq_laneq_u32_0(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmulq_laneq_u32_0:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> zeroinitializer
  %mul = mul <4 x i32> %shuffle, %a
  ret <4 x i32> %mul
}

define <2 x float> @test_vfma_lane_f32_0(<2 x float> %a, <2 x float> %b, <2 x float> %v) {
; CHECK: test_vfma_lane_f32_0:
; CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %lane = shufflevector <2 x float> %v, <2 x float> undef, <2 x i32> zeroinitializer
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %lane, <2 x float> %b, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmaq_lane_f32_0(<4 x float> %a, <4 x float> %b, <2 x float> %v) {
; CHECK: test_vfmaq_lane_f32_0:
; CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %lane = shufflevector <2 x float> %v, <2 x float> undef, <4 x i32> zeroinitializer
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %lane, <4 x float> %b, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x float> @test_vfma_laneq_f32_0(<2 x float> %a, <2 x float> %b, <4 x float> %v) {
; CHECK: test_vfma_laneq_f32_0:
; CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %lane = shufflevector <4 x float> %v, <4 x float> undef, <2 x i32> zeroinitializer
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %lane, <2 x float> %b, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmaq_laneq_f32_0(<4 x float> %a, <4 x float> %b, <4 x float> %v) {
; CHECK: test_vfmaq_laneq_f32_0:
; CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %lane = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> zeroinitializer
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %lane, <4 x float> %b, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x float> @test_vfms_lane_f32_0(<2 x float> %a, <2 x float> %b, <2 x float> %v) {
; CHECK: test_vfms_lane_f32_0:
; CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %sub = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %v
  %lane = shufflevector <2 x float> %sub, <2 x float> undef, <2 x i32> zeroinitializer
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %lane, <2 x float> %b, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmsq_lane_f32_0(<4 x float> %a, <4 x float> %b, <2 x float> %v) {
; CHECK: test_vfmsq_lane_f32_0:
; CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %sub = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %v
  %lane = shufflevector <2 x float> %sub, <2 x float> undef, <4 x i32> zeroinitializer
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %lane, <4 x float> %b, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x float> @test_vfms_laneq_f32_0(<2 x float> %a, <2 x float> %b, <4 x float> %v) {
; CHECK: test_vfms_laneq_f32_0:
; CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %sub = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %v
  %lane = shufflevector <4 x float> %sub, <4 x float> undef, <2 x i32> zeroinitializer
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %lane, <2 x float> %b, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmsq_laneq_f32_0(<4 x float> %a, <4 x float> %b, <4 x float> %v) {
; CHECK: test_vfmsq_laneq_f32_0:
; CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %sub = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %v
  %lane = shufflevector <4 x float> %sub, <4 x float> undef, <4 x i32> zeroinitializer
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %lane, <4 x float> %b, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x double> @test_vfmaq_laneq_f64_0(<2 x double> %a, <2 x double> %b, <2 x double> %v) {
; CHECK: test_vfmaq_laneq_f64_0:
; CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %lane = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> zeroinitializer
  %0 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %lane, <2 x double> %b, <2 x double> %a)
  ret <2 x double> %0
}

define <2 x double> @test_vfmsq_laneq_f64_0(<2 x double> %a, <2 x double> %b, <2 x double> %v) {
; CHECK: test_vfmsq_laneq_f64_0:
; CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %sub = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %v
  %lane = shufflevector <2 x double> %sub, <2 x double> undef, <2 x i32> zeroinitializer
  %0 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %lane, <2 x double> %b, <2 x double> %a)
  ret <2 x double> %0
}

define <4 x i32> @test_vmlal_lane_s16_0(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlal_lane_s16_0:
; CHECK: mlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_lane_s32_0(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlal_lane_s32_0:
; CHECK: mlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_laneq_s16_0(<4 x i32> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlal_laneq_s16_0:
; CHECK: mlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_laneq_s32_0(<2 x i64> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlal_laneq_s32_0:
; CHECK: mlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_high_lane_s16_0(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlal_high_lane_s16_0:
; CHECK: mlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_high_lane_s32_0(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlal_high_lane_s32_0:
; CHECK: mlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_high_laneq_s16_0(<4 x i32> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlal_high_laneq_s16_0:
; CHECK: mlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_high_laneq_s32_0(<2 x i64> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlal_high_laneq_s32_0:
; CHECK: mlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlsl_lane_s16_0(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsl_lane_s16_0:
; CHECK: mlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_lane_s32_0(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsl_lane_s32_0:
; CHECK: mlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_laneq_s16_0(<4 x i32> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsl_laneq_s16_0:
; CHECK: mlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_laneq_s32_0(<2 x i64> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsl_laneq_s32_0:
; CHECK: mlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_high_lane_s16_0(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsl_high_lane_s16_0:
; CHECK: mlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_high_lane_s32_0(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsl_high_lane_s32_0:
; CHECK: mlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_high_laneq_s16_0(<4 x i32> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsl_high_laneq_s16_0:
; CHECK: mlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_high_laneq_s32_0(<2 x i64> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsl_high_laneq_s32_0:
; CHECK: mlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlal_lane_u16_0(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlal_lane_u16_0:
; CHECK: mlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_lane_u32_0(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlal_lane_u32_0:
; CHECK: mlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_laneq_u16_0(<4 x i32> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlal_laneq_u16_0:
; CHECK: mlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_laneq_u32_0(<2 x i64> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlal_laneq_u32_0:
; CHECK: mlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_high_lane_u16_0(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlal_high_lane_u16_0:
; CHECK: mlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_high_lane_u32_0(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlal_high_lane_u32_0:
; CHECK: mlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlal_high_laneq_u16_0(<4 x i32> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlal_high_laneq_u16_0:
; CHECK: mlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %add = add <4 x i32> %vmull2.i, %a
  ret <4 x i32> %add
}

define <2 x i64> @test_vmlal_high_laneq_u32_0(<2 x i64> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlal_high_laneq_u32_0:
; CHECK: mlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %add = add <2 x i64> %vmull2.i, %a
  ret <2 x i64> %add
}

define <4 x i32> @test_vmlsl_lane_u16_0(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsl_lane_u16_0:
; CHECK: mlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_lane_u32_0(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsl_lane_u32_0:
; CHECK: mlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_laneq_u16_0(<4 x i32> %a, <4 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsl_laneq_u16_0:
; CHECK: mlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_laneq_u32_0(<2 x i64> %a, <2 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsl_laneq_u32_0:
; CHECK: mlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_high_lane_u16_0(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vmlsl_high_lane_u16_0:
; CHECK: mlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_high_lane_u32_0(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vmlsl_high_lane_u32_0:
; CHECK: mlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmlsl_high_laneq_u16_0(<4 x i32> %a, <8 x i16> %b, <8 x i16> %v) {
; CHECK: test_vmlsl_high_laneq_u16_0:
; CHECK: mlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %sub = sub <4 x i32> %a, %vmull2.i
  ret <4 x i32> %sub
}

define <2 x i64> @test_vmlsl_high_laneq_u32_0(<2 x i64> %a, <4 x i32> %b, <4 x i32> %v) {
; CHECK: test_vmlsl_high_laneq_u32_0:
; CHECK: mlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %sub = sub <2 x i64> %a, %vmull2.i
  ret <2 x i64> %sub
}

define <4 x i32> @test_vmull_lane_s16_0(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmull_lane_s16_0:
; CHECK: mull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_lane_s32_0(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmull_lane_s32_0:
; CHECK: mull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_lane_u16_0(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmull_lane_u16_0:
; CHECK: mull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_lane_u32_0(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmull_lane_u32_0:
; CHECK: mull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_high_lane_s16_0(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmull_high_lane_s16_0:
; CHECK: mull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_high_lane_s32_0(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmull_high_lane_s32_0:
; CHECK: mull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_high_lane_u16_0(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vmull_high_lane_u16_0:
; CHECK: mull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_high_lane_u32_0(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vmull_high_lane_u32_0:
; CHECK: mull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_laneq_s16_0(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmull_laneq_s16_0:
; CHECK: mull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_laneq_s32_0(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmull_laneq_s32_0:
; CHECK: mull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_laneq_u16_0(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmull_laneq_u16_0:
; CHECK: mull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_laneq_u32_0(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmull_laneq_u32_0:
; CHECK: mull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_high_laneq_s16_0(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmull_high_laneq_s16_0:
; CHECK: mull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_high_laneq_s32_0(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmull_high_laneq_s32_0:
; CHECK: mull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vmull_high_laneq_u16_0(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vmull_high_laneq_u16_0:
; CHECK: mull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vmull2.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_high_laneq_u32_0(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vmull_high_laneq_u32_0:
; CHECK: mull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vmull2.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vmull2.i
}

define <4 x i32> @test_vqdmlal_lane_s16_0(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vqdmlal_lane_s16_0:
; CHECK: qdmlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vqdmlal2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %vqdmlal4.i = tail call <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal2.i)
  ret <4 x i32> %vqdmlal4.i
}

define <2 x i64> @test_vqdmlal_lane_s32_0(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vqdmlal_lane_s32_0:
; CHECK: qdmlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vqdmlal2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %vqdmlal4.i = tail call <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal2.i)
  ret <2 x i64> %vqdmlal4.i
}

define <4 x i32> @test_vqdmlal_high_lane_s16_0(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vqdmlal_high_lane_s16_0:
; CHECK: qdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vqdmlal2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %vqdmlal4.i = tail call <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal2.i)
  ret <4 x i32> %vqdmlal4.i
}

define <2 x i64> @test_vqdmlal_high_lane_s32_0(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vqdmlal_high_lane_s32_0:
; CHECK: qdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vqdmlal2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %vqdmlal4.i = tail call <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal2.i)
  ret <2 x i64> %vqdmlal4.i
}

define <4 x i32> @test_vqdmlsl_lane_s16_0(<4 x i32> %a, <4 x i16> %b, <4 x i16> %v) {
; CHECK: test_vqdmlsl_lane_s16_0:
; CHECK: qdmlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vqdmlsl2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %b, <4 x i16> %shuffle)
  %vqdmlsl4.i = tail call <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl2.i)
  ret <4 x i32> %vqdmlsl4.i
}

define <2 x i64> @test_vqdmlsl_lane_s32_0(<2 x i64> %a, <2 x i32> %b, <2 x i32> %v) {
; CHECK: test_vqdmlsl_lane_s32_0:
; CHECK: qdmlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vqdmlsl2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %b, <2 x i32> %shuffle)
  %vqdmlsl4.i = tail call <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl2.i)
  ret <2 x i64> %vqdmlsl4.i
}

define <4 x i32> @test_vqdmlsl_high_lane_s16_0(<4 x i32> %a, <8 x i16> %b, <4 x i16> %v) {
; CHECK: test_vqdmlsl_high_lane_s16_0:
; CHECK: qdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vqdmlsl2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  %vqdmlsl4.i = tail call <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl2.i)
  ret <4 x i32> %vqdmlsl4.i
}

define <2 x i64> @test_vqdmlsl_high_lane_s32_0(<2 x i64> %a, <4 x i32> %b, <2 x i32> %v) {
; CHECK: test_vqdmlsl_high_lane_s32_0:
; CHECK: qdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vqdmlsl2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  %vqdmlsl4.i = tail call <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl2.i)
  ret <2 x i64> %vqdmlsl4.i
}

define <4 x i32> @test_vqdmull_lane_s16_0(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqdmull_lane_s16_0:
; CHECK: qdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vqdmull2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_lane_s32_0(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqdmull_lane_s32_0:
; CHECK: qdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vqdmull2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i32> @test_vqdmull_laneq_s16_0(<4 x i16> %a, <8 x i16> %v) {
; CHECK: test_vqdmull_laneq_s16_0:
; CHECK: qdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vqdmull2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_laneq_s32_0(<2 x i32> %a, <4 x i32> %v) {
; CHECK: test_vqdmull_laneq_s32_0:
; CHECK: qdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vqdmull2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i32> @test_vqdmull_high_lane_s16_0(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqdmull_high_lane_s16_0:
; CHECK: qdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vqdmull2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_high_lane_s32_0(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqdmull_high_lane_s32_0:
; CHECK: qdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vqdmull2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i32> @test_vqdmull_high_laneq_s16_0(<8 x i16> %a, <8 x i16> %v) {
; CHECK: test_vqdmull_high_laneq_s16_0:
; CHECK: qdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> zeroinitializer
  %vqdmull2.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i, <4 x i16> %shuffle)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_high_laneq_s32_0(<4 x i32> %a, <4 x i32> %v) {
; CHECK: test_vqdmull_high_laneq_s32_0:
; CHECK: qdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> undef, <2 x i32> zeroinitializer
  %vqdmull2.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i, <2 x i32> %shuffle)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i16> @test_vqdmulh_lane_s16_0(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqdmulh_lane_s16_0:
; CHECK: qdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vqdmulh2.i = tail call <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i16> %vqdmulh2.i
}

define <8 x i16> @test_vqdmulhq_lane_s16_0(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqdmulhq_lane_s16_0:
; CHECK: qdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> zeroinitializer
  %vqdmulh2.i = tail call <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16> %a, <8 x i16> %shuffle)
  ret <8 x i16> %vqdmulh2.i
}

define <2 x i32> @test_vqdmulh_lane_s32_0(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqdmulh_lane_s32_0:
; CHECK: qdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vqdmulh2.i = tail call <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i32> %vqdmulh2.i
}

define <4 x i32> @test_vqdmulhq_lane_s32_0(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqdmulhq_lane_s32_0:
; CHECK: qdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> zeroinitializer
  %vqdmulh2.i = tail call <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32> %a, <4 x i32> %shuffle)
  ret <4 x i32> %vqdmulh2.i
}

define <4 x i16> @test_vqrdmulh_lane_s16_0(<4 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqrdmulh_lane_s16_0:
; CHECK: qrdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> zeroinitializer
  %vqrdmulh2.i = tail call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %a, <4 x i16> %shuffle)
  ret <4 x i16> %vqrdmulh2.i
}

define <8 x i16> @test_vqrdmulhq_lane_s16_0(<8 x i16> %a, <4 x i16> %v) {
; CHECK: test_vqrdmulhq_lane_s16_0:
; CHECK: qrdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> zeroinitializer
  %vqrdmulh2.i = tail call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %a, <8 x i16> %shuffle)
  ret <8 x i16> %vqrdmulh2.i
}

define <2 x i32> @test_vqrdmulh_lane_s32_0(<2 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqrdmulh_lane_s32_0:
; CHECK: qrdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> zeroinitializer
  %vqrdmulh2.i = tail call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %a, <2 x i32> %shuffle)
  ret <2 x i32> %vqrdmulh2.i
}

define <4 x i32> @test_vqrdmulhq_lane_s32_0(<4 x i32> %a, <2 x i32> %v) {
; CHECK: test_vqrdmulhq_lane_s32_0:
; CHECK: qrdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> zeroinitializer
  %vqrdmulh2.i = tail call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %a, <4 x i32> %shuffle)
  ret <4 x i32> %vqrdmulh2.i
}

define <2 x float> @test_vmul_lane_f32_0(<2 x float> %a, <2 x float> %v) {
; CHECK: test_vmul_lane_f32_0:
; CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <2 x i32> zeroinitializer
  %mul = fmul <2 x float> %shuffle, %a
  ret <2 x float> %mul
}

define <4 x float> @test_vmulq_lane_f32_0(<4 x float> %a, <2 x float> %v) {
; CHECK: test_vmulq_lane_f32_0:
; CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <4 x i32> zeroinitializer
  %mul = fmul <4 x float> %shuffle, %a
  ret <4 x float> %mul
}

define <2 x float> @test_vmul_laneq_f32_0(<2 x float> %a, <4 x float> %v) {
; CHECK: test_vmul_laneq_f32_0:
; CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x float> %v, <4 x float> undef, <2 x i32> zeroinitializer
  %mul = fmul <2 x float> %shuffle, %a
  ret <2 x float> %mul
}

define <1 x double> @test_vmul_laneq_f64_0(<1 x double> %a, <2 x double> %v) {
; CHECK: test_vmul_laneq_f64_0:
; CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
entry:
  %0 = bitcast <1 x double> %a to <8 x i8>
  %1 = bitcast <8 x i8> %0 to double
  %extract = extractelement <2 x double> %v, i32 0
  %2 = fmul double %1, %extract
  %3 = insertelement <1 x double> undef, double %2, i32 0
  ret <1 x double> %3
}

define <4 x float> @test_vmulq_laneq_f32_0(<4 x float> %a, <4 x float> %v) {
; CHECK: test_vmulq_laneq_f32_0:
; CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> zeroinitializer
  %mul = fmul <4 x float> %shuffle, %a
  ret <4 x float> %mul
}

define <2 x double> @test_vmulq_laneq_f64_0(<2 x double> %a, <2 x double> %v) {
; CHECK: test_vmulq_laneq_f64_0:
; CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %shuffle = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> zeroinitializer
  %mul = fmul <2 x double> %shuffle, %a
  ret <2 x double> %mul
}

define <2 x float> @test_vmulx_lane_f32_0(<2 x float> %a, <2 x float> %v) {
; CHECK: test_vmulx_lane_f32_0:
; CHECK: mulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <2 x i32> zeroinitializer
  %vmulx2.i = tail call <2 x float> @llvm.aarch64.neon.vmulx.v2f32(<2 x float> %a, <2 x float> %shuffle)
  ret <2 x float> %vmulx2.i
}

define <4 x float> @test_vmulxq_lane_f32_0(<4 x float> %a, <2 x float> %v) {
; CHECK: test_vmulxq_lane_f32_0:
; CHECK: mulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <4 x i32> zeroinitializer
  %vmulx2.i = tail call <4 x float> @llvm.aarch64.neon.vmulx.v4f32(<4 x float> %a, <4 x float> %shuffle)
  ret <4 x float> %vmulx2.i
}

define <2 x double> @test_vmulxq_lane_f64_0(<2 x double> %a, <1 x double> %v) {
; CHECK: test_vmulxq_lane_f64_0:
; CHECK: mulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %shuffle = shufflevector <1 x double> %v, <1 x double> undef, <2 x i32> zeroinitializer
  %vmulx2.i = tail call <2 x double> @llvm.aarch64.neon.vmulx.v2f64(<2 x double> %a, <2 x double> %shuffle)
  ret <2 x double> %vmulx2.i
}

define <2 x float> @test_vmulx_laneq_f32_0(<2 x float> %a, <4 x float> %v) {
; CHECK: test_vmulx_laneq_f32_0:
; CHECK: mulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x float> %v, <4 x float> undef, <2 x i32> zeroinitializer
  %vmulx2.i = tail call <2 x float> @llvm.aarch64.neon.vmulx.v2f32(<2 x float> %a, <2 x float> %shuffle)
  ret <2 x float> %vmulx2.i
}

define <4 x float> @test_vmulxq_laneq_f32_0(<4 x float> %a, <4 x float> %v) {
; CHECK: test_vmulxq_laneq_f32_0:
; CHECK: mulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> zeroinitializer
  %vmulx2.i = tail call <4 x float> @llvm.aarch64.neon.vmulx.v4f32(<4 x float> %a, <4 x float> %shuffle)
  ret <4 x float> %vmulx2.i
}

define <2 x double> @test_vmulxq_laneq_f64_0(<2 x double> %a, <2 x double> %v) {
; CHECK: test_vmulxq_laneq_f64_0:
; CHECK: mulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %shuffle = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> zeroinitializer
  %vmulx2.i = tail call <2 x double> @llvm.aarch64.neon.vmulx.v2f64(<2 x double> %a, <2 x double> %shuffle)
  ret <2 x double> %vmulx2.i
}

