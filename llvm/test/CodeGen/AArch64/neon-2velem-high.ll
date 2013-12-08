; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s

declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)

declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)

declare <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32>, <2 x i32>)

declare <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64>, <2 x i64>)

declare <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16>, <4 x i16>)

declare <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64>, <2 x i64>)

declare <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32>, <2 x i32>)

declare <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16>, <4 x i16>)

declare <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32>, <2 x i32>)

declare <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16>, <4 x i16>)

define <4 x i32> @test_vmull_high_n_s16(<8 x i16> %a, i16 %b) {
; CHECK: test_vmull_high_n_s16:
; CHECK: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %b, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %b, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %b, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %b, i32 3
  %vmull15.i.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  ret <4 x i32> %vmull15.i.i
}

define <2 x i64> @test_vmull_high_n_s32(<4 x i32> %a, i32 %b) {
; CHECK: test_vmull_high_n_s32:
; CHECK: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %b, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %b, i32 1
  %vmull9.i.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  ret <2 x i64> %vmull9.i.i
}

define <4 x i32> @test_vmull_high_n_u16(<8 x i16> %a, i16 %b) {
; CHECK: test_vmull_high_n_u16:
; CHECK: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %b, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %b, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %b, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %b, i32 3
  %vmull15.i.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  ret <4 x i32> %vmull15.i.i
}

define <2 x i64> @test_vmull_high_n_u32(<4 x i32> %a, i32 %b) {
; CHECK: test_vmull_high_n_u32:
; CHECK: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %b, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %b, i32 1
  %vmull9.i.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  ret <2 x i64> %vmull9.i.i
}

define <4 x i32> @test_vqdmull_high_n_s16(<8 x i16> %a, i16 %b) {
; CHECK: test_vqdmull_high_n_s16:
; CHECK: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %b, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %b, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %b, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %b, i32 3
  %vqdmull15.i.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  ret <4 x i32> %vqdmull15.i.i
}

define <2 x i64> @test_vqdmull_high_n_s32(<4 x i32> %a, i32 %b) {
; CHECK: test_vqdmull_high_n_s32:
; CHECK: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %b, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %b, i32 1
  %vqdmull9.i.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  ret <2 x i64> %vqdmull9.i.i
}

define <4 x i32> @test_vmlal_high_n_s16(<4 x i32> %a, <8 x i16> %b, i16 %c) {
; CHECK: test_vmlal_high_n_s16:
; CHECK: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vmull2.i.i.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %add.i.i = add <4 x i32> %vmull2.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <2 x i64> @test_vmlal_high_n_s32(<2 x i64> %a, <4 x i32> %b, i32 %c) {
; CHECK: test_vmlal_high_n_s32:
; CHECK: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vmull2.i.i.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %add.i.i = add <2 x i64> %vmull2.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <4 x i32> @test_vmlal_high_n_u16(<4 x i32> %a, <8 x i16> %b, i16 %c) {
; CHECK: test_vmlal_high_n_u16:
; CHECK: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vmull2.i.i.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %add.i.i = add <4 x i32> %vmull2.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <2 x i64> @test_vmlal_high_n_u32(<2 x i64> %a, <4 x i32> %b, i32 %c) {
; CHECK: test_vmlal_high_n_u32:
; CHECK: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vmull2.i.i.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %add.i.i = add <2 x i64> %vmull2.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <4 x i32> @test_vqdmlal_high_n_s16(<4 x i32> %a, <8 x i16> %b, i16 %c) {
; CHECK: test_vqdmlal_high_n_s16:
; CHECK: sqdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vqdmlal15.i.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %vqdmlal17.i.i = tail call <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal15.i.i)
  ret <4 x i32> %vqdmlal17.i.i
}

define <2 x i64> @test_vqdmlal_high_n_s32(<2 x i64> %a, <4 x i32> %b, i32 %c) {
; CHECK: test_vqdmlal_high_n_s32:
; CHECK: sqdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vqdmlal9.i.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %vqdmlal11.i.i = tail call <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal9.i.i)
  ret <2 x i64> %vqdmlal11.i.i
}

define <4 x i32> @test_vmlsl_high_n_s16(<4 x i32> %a, <8 x i16> %b, i16 %c) {
; CHECK: test_vmlsl_high_n_s16:
; CHECK: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vmull2.i.i.i = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %sub.i.i = sub <4 x i32> %a, %vmull2.i.i.i
  ret <4 x i32> %sub.i.i
}

define <2 x i64> @test_vmlsl_high_n_s32(<2 x i64> %a, <4 x i32> %b, i32 %c) {
; CHECK: test_vmlsl_high_n_s32:
; CHECK: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vmull2.i.i.i = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %sub.i.i = sub <2 x i64> %a, %vmull2.i.i.i
  ret <2 x i64> %sub.i.i
}

define <4 x i32> @test_vmlsl_high_n_u16(<4 x i32> %a, <8 x i16> %b, i16 %c) {
; CHECK: test_vmlsl_high_n_u16:
; CHECK: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vmull2.i.i.i = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %sub.i.i = sub <4 x i32> %a, %vmull2.i.i.i
  ret <4 x i32> %sub.i.i
}

define <2 x i64> @test_vmlsl_high_n_u32(<2 x i64> %a, <4 x i32> %b, i32 %c) {
; CHECK: test_vmlsl_high_n_u32:
; CHECK: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vmull2.i.i.i = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %sub.i.i = sub <2 x i64> %a, %vmull2.i.i.i
  ret <2 x i64> %sub.i.i
}

define <4 x i32> @test_vqdmlsl_high_n_s16(<4 x i32> %a, <8 x i16> %b, i16 %c) {
; CHECK: test_vqdmlsl_high_n_s16:
; CHECK: sqdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vqdmlsl15.i.i = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %vqdmlsl17.i.i = tail call <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl15.i.i)
  ret <4 x i32> %vqdmlsl17.i.i
}

define <2 x i64> @test_vqdmlsl_high_n_s32(<2 x i64> %a, <4 x i32> %b, i32 %c) {
; CHECK: test_vqdmlsl_high_n_s32:
; CHECK: sqdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vqdmlsl9.i.i = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %vqdmlsl11.i.i = tail call <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl9.i.i)
  ret <2 x i64> %vqdmlsl11.i.i
}

define <2 x float> @test_vmul_n_f32(<2 x float> %a, float %b) {
; CHECK: test_vmul_n_f32:
; CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %vecinit.i = insertelement <2 x float> undef, float %b, i32 0
  %vecinit1.i = insertelement <2 x float> %vecinit.i, float %b, i32 1
  %mul.i = fmul <2 x float> %vecinit1.i, %a
  ret <2 x float> %mul.i
}

define <4 x float> @test_vmulq_n_f32(<4 x float> %a, float %b) {
; CHECK: test_vmulq_n_f32:
; CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
entry:
  %vecinit.i = insertelement <4 x float> undef, float %b, i32 0
  %vecinit1.i = insertelement <4 x float> %vecinit.i, float %b, i32 1
  %vecinit2.i = insertelement <4 x float> %vecinit1.i, float %b, i32 2
  %vecinit3.i = insertelement <4 x float> %vecinit2.i, float %b, i32 3
  %mul.i = fmul <4 x float> %vecinit3.i, %a
  ret <4 x float> %mul.i
}

define <2 x double> @test_vmulq_n_f64(<2 x double> %a, double %b) {
; CHECK: test_vmulq_n_f64:
; CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
entry:
  %vecinit.i = insertelement <2 x double> undef, double %b, i32 0
  %vecinit1.i = insertelement <2 x double> %vecinit.i, double %b, i32 1
  %mul.i = fmul <2 x double> %vecinit1.i, %a
  ret <2 x double> %mul.i
}

define <2 x float> @test_vfma_n_f32(<2 x float> %a, <2 x float> %b, float %n) {
; CHECK: test_vfma_n_f32:
; CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %vecinit.i = insertelement <2 x float> undef, float %n, i32 0
  %vecinit1.i = insertelement <2 x float> %vecinit.i, float %n, i32 1
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %b, <2 x float> %vecinit1.i, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmaq_n_f32(<4 x float> %a, <4 x float> %b, float %n) {
; CHECK: test_vfmaq_n_f32:
; CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %vecinit.i = insertelement <4 x float> undef, float %n, i32 0
  %vecinit1.i = insertelement <4 x float> %vecinit.i, float %n, i32 1
  %vecinit2.i = insertelement <4 x float> %vecinit1.i, float %n, i32 2
  %vecinit3.i = insertelement <4 x float> %vecinit2.i, float %n, i32 3
  %0 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %b, <4 x float> %vecinit3.i, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x float> @test_vfms_n_f32(<2 x float> %a, <2 x float> %b, float %n) {
; CHECK: test_vfms_n_f32:
; CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %vecinit.i = insertelement <2 x float> undef, float %n, i32 0
  %vecinit1.i = insertelement <2 x float> %vecinit.i, float %n, i32 1
  %0 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %b
  %1 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %0, <2 x float> %vecinit1.i, <2 x float> %a)
  ret <2 x float> %1
}

define <4 x float> @test_vfmsq_n_f32(<4 x float> %a, <4 x float> %b, float %n) {
; CHECK: test_vfmsq_n_f32:
; CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
entry:
  %vecinit.i = insertelement <4 x float> undef, float %n, i32 0
  %vecinit1.i = insertelement <4 x float> %vecinit.i, float %n, i32 1
  %vecinit2.i = insertelement <4 x float> %vecinit1.i, float %n, i32 2
  %vecinit3.i = insertelement <4 x float> %vecinit2.i, float %n, i32 3
  %0 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %b
  %1 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %0, <4 x float> %vecinit3.i, <4 x float> %a)
  ret <4 x float> %1
}
