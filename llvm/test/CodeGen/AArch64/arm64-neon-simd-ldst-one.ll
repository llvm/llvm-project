; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon | FileCheck %s


%struct.uint8x16x2_t = type { [2 x <16 x i8>] }
%struct.poly8x16x2_t = type { [2 x <16 x i8>] }
%struct.uint8x16x3_t = type { [3 x <16 x i8>] }
%struct.int8x16x2_t = type { [2 x <16 x i8>] }
%struct.int16x8x2_t = type { [2 x <8 x i16>] }
%struct.int32x4x2_t = type { [2 x <4 x i32>] }
%struct.int64x2x2_t = type { [2 x <2 x i64>] }
%struct.float32x4x2_t = type { [2 x <4 x float>] }
%struct.float64x2x2_t = type { [2 x <2 x double>] }
%struct.int8x8x2_t = type { [2 x <8 x i8>] }
%struct.int16x4x2_t = type { [2 x <4 x i16>] }
%struct.int32x2x2_t = type { [2 x <2 x i32>] }
%struct.int64x1x2_t = type { [2 x <1 x i64>] }
%struct.float32x2x2_t = type { [2 x <2 x float>] }
%struct.float64x1x2_t = type { [2 x <1 x double>] }
%struct.int8x16x3_t = type { [3 x <16 x i8>] }
%struct.int16x8x3_t = type { [3 x <8 x i16>] }
%struct.int32x4x3_t = type { [3 x <4 x i32>] }
%struct.int64x2x3_t = type { [3 x <2 x i64>] }
%struct.float32x4x3_t = type { [3 x <4 x float>] }
%struct.float64x2x3_t = type { [3 x <2 x double>] }
%struct.int8x8x3_t = type { [3 x <8 x i8>] }
%struct.int16x4x3_t = type { [3 x <4 x i16>] }
%struct.int32x2x3_t = type { [3 x <2 x i32>] }
%struct.int64x1x3_t = type { [3 x <1 x i64>] }
%struct.float32x2x3_t = type { [3 x <2 x float>] }
%struct.float64x1x3_t = type { [3 x <1 x double>] }
%struct.int8x16x4_t = type { [4 x <16 x i8>] }
%struct.int16x8x4_t = type { [4 x <8 x i16>] }
%struct.int32x4x4_t = type { [4 x <4 x i32>] }
%struct.int64x2x4_t = type { [4 x <2 x i64>] }
%struct.float32x4x4_t = type { [4 x <4 x float>] }
%struct.float64x2x4_t = type { [4 x <2 x double>] }
%struct.int8x8x4_t = type { [4 x <8 x i8>] }
%struct.int16x4x4_t = type { [4 x <4 x i16>] }
%struct.int32x2x4_t = type { [4 x <2 x i32>] }
%struct.int64x1x4_t = type { [4 x <1 x i64>] }
%struct.float32x2x4_t = type { [4 x <2 x float>] }
%struct.float64x1x4_t = type { [4 x <1 x double>] }

define <16 x i8> @test_ld_from_poll_v16i8(<16 x i8> %a) {
; CHECK-LABEL: test_ld_from_poll_v16i8:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <16 x i8> %a, <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 2, i8 13, i8 14, i8 15, i8 16>
  ret <16 x i8> %b
}

define <8 x i16> @test_ld_from_poll_v8i16(<8 x i16> %a) {
; CHECK-LABEL: test_ld_from_poll_v8i16:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <8 x i16> %a, <i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8>
  ret <8 x i16> %b
}

define <4 x i32> @test_ld_from_poll_v4i32(<4 x i32> %a) {
; CHECK-LABEL: test_ld_from_poll_v4i32:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <4 x i32> %a, <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i32> %b
}

define <2 x i64> @test_ld_from_poll_v2i64(<2 x i64> %a) {
; CHECK-LABEL: test_ld_from_poll_v2i64:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <2 x i64> %a, <i64 1, i64 2>
  ret <2 x i64> %b
}

define <4 x float> @test_ld_from_poll_v4f32(<4 x float> %a) {
; CHECK-LABEL: test_ld_from_poll_v4f32:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = fadd <4 x float> %a, <float 1.0, float 2.0, float 3.0, float 4.0>
  ret <4 x float> %b
}

define <2 x double> @test_ld_from_poll_v2f64(<2 x double> %a) {
; CHECK-LABEL: test_ld_from_poll_v2f64:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = fadd <2 x double> %a, <double 1.0, double 2.0>
  ret <2 x double> %b
}

define <8 x i8> @test_ld_from_poll_v8i8(<8 x i8> %a) {
; CHECK-LABEL: test_ld_from_poll_v8i8:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <8 x i8> %a, <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>
  ret <8 x i8> %b
}

define <4 x i16> @test_ld_from_poll_v4i16(<4 x i16> %a) {
; CHECK-LABEL: test_ld_from_poll_v4i16:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <4 x i16> %a, <i16 1, i16 2, i16 3, i16 4>
  ret <4 x i16> %b
}

define <2 x i32> @test_ld_from_poll_v2i32(<2 x i32> %a) {
; CHECK-LABEL: test_ld_from_poll_v2i32:
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <2 x i32> %a, <i32 1, i32 2>
  ret <2 x i32> %b
}

define <16 x i8> @test_vld1q_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld1q_dup_s8:
; CHECK: ld1r {{{ ?v[0-9]+.16b ?}}}, [x0]
entry:
  %0 = load i8, i8* %a, align 1
  %1 = insertelement <16 x i8> undef, i8 %0, i32 0
  %lane = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %lane
}

define <8 x i16> @test_vld1q_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld1q_dup_s16:
; CHECK: ld1r {{{ ?v[0-9]+.8h ?}}}, [x0]
entry:
  %0 = load i16, i16* %a, align 2
  %1 = insertelement <8 x i16> undef, i16 %0, i32 0
  %lane = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %lane
}

define <4 x i32> @test_vld1q_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld1q_dup_s32:
; CHECK: ld1r {{{ ?v[0-9]+.4s ?}}}, [x0]
entry:
  %0 = load i32, i32* %a, align 4
  %1 = insertelement <4 x i32> undef, i32 %0, i32 0
  %lane = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %lane
}

define <2 x i64> @test_vld1q_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld1q_dup_s64:
; CHECK: ld1r {{{ ?v[0-9]+.2d ?}}}, [x0]
entry:
  %0 = load i64, i64* %a, align 8
  %1 = insertelement <2 x i64> undef, i64 %0, i32 0
  %lane = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %lane
}

define <4 x float> @test_vld1q_dup_f32(float* %a) {
; CHECK-LABEL: test_vld1q_dup_f32:
; CHECK: ld1r {{{ ?v[0-9]+.4s ?}}}, [x0]
entry:
  %0 = load float, float* %a, align 4
  %1 = insertelement <4 x float> undef, float %0, i32 0
  %lane = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %lane
}

define <2 x double> @test_vld1q_dup_f64(double* %a) {
; CHECK-LABEL: test_vld1q_dup_f64:
; CHECK: ld1r {{{ ?v[0-9]+.2d ?}}}, [x0]
entry:
  %0 = load double, double* %a, align 8
  %1 = insertelement <2 x double> undef, double %0, i32 0
  %lane = shufflevector <2 x double> %1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %lane
}

define <8 x i8> @test_vld1_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld1_dup_s8:
; CHECK: ld1r {{{ ?v[0-9]+.8b ?}}}, [x0]
entry:
  %0 = load i8, i8* %a, align 1
  %1 = insertelement <8 x i8> undef, i8 %0, i32 0
  %lane = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  ret <8 x i8> %lane
}

define <4 x i16> @test_vld1_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld1_dup_s16:
; CHECK: ld1r {{{ ?v[0-9]+.4h ?}}}, [x0]
entry:
  %0 = load i16, i16* %a, align 2
  %1 = insertelement <4 x i16> undef, i16 %0, i32 0
  %lane = shufflevector <4 x i16> %1, <4 x i16> undef, <4 x i32> zeroinitializer
  ret <4 x i16> %lane
}

define <2 x i32> @test_vld1_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld1_dup_s32:
; CHECK: ld1r {{{ ?v[0-9]+.2s ?}}}, [x0]
entry:
  %0 = load i32, i32* %a, align 4
  %1 = insertelement <2 x i32> undef, i32 %0, i32 0
  %lane = shufflevector <2 x i32> %1, <2 x i32> undef, <2 x i32> zeroinitializer
  ret <2 x i32> %lane
}

define <1 x i64> @test_vld1_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld1_dup_s64:
; CHECK: ldr {{d[0-9]+}}, [x0]
entry:
  %0 = load i64, i64* %a, align 8
  %1 = insertelement <1 x i64> undef, i64 %0, i32 0
  ret <1 x i64> %1
}

define <2 x float> @test_vld1_dup_f32(float* %a) {
; CHECK-LABEL: test_vld1_dup_f32:
; CHECK: ld1r {{{ ?v[0-9]+.2s ?}}}, [x0]
entry:
  %0 = load float, float* %a, align 4
  %1 = insertelement <2 x float> undef, float %0, i32 0
  %lane = shufflevector <2 x float> %1, <2 x float> undef, <2 x i32> zeroinitializer
  ret <2 x float> %lane
}

define <1 x double> @test_vld1_dup_f64(double* %a) {
; CHECK-LABEL: test_vld1_dup_f64:
; CHECK: ldr {{d[0-9]+}}, [x0]
entry:
  %0 = load double, double* %a, align 8
  %1 = insertelement <1 x double> undef, double %0, i32 0
  ret <1 x double> %1
}

define <1 x i64> @testDUP.v1i64(i64* %a, i64* %b) #0 {
; As there is a store operation depending on %1, LD1R pattern can't be selected.
; So LDR and FMOV should be emitted.
; CHECK-LABEL: testDUP.v1i64:
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}]
; CHECK-DAG: fmov {{d[0-9]+}}, {{x[0-9]+}}
; CHECK-DAG: str {{x[0-9]+}}, [{{x[0-9]+}}]
  %1 = load i64, i64* %a, align 8
  store i64 %1, i64* %b, align 8
  %vecinit.i = insertelement <1 x i64> undef, i64 %1, i32 0
  ret <1 x i64> %vecinit.i
}

define <1 x double> @testDUP.v1f64(double* %a, double* %b) #0 {
; As there is a store operation depending on %1, LD1R pattern can't be selected.
; So LDR and FMOV should be emitted.
; CHECK-LABEL: testDUP.v1f64:
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}]
; CHECK: str {{d[0-9]+}}, [{{x[0-9]+}}]
  %1 = load double, double* %a, align 8
  store double %1, double* %b, align 8
  %vecinit.i = insertelement <1 x double> undef, double %1, i32 0
  ret <1 x double> %vecinit.i
}

define <16 x i8> @test_vld1q_lane_s8(i8* %a, <16 x i8> %b) {
; CHECK-LABEL: test_vld1q_lane_s8:
; CHECK: ld1 { {{v[0-9]+}}.b }[{{[0-9]+}}], [x0]
entry:
  %0 = load i8, i8* %a, align 1
  %vld1_lane = insertelement <16 x i8> %b, i8 %0, i32 15
  ret <16 x i8> %vld1_lane
}

define <8 x i16> @test_vld1q_lane_s16(i16* %a, <8 x i16> %b) {
; CHECK-LABEL: test_vld1q_lane_s16:
; CHECK: ld1 { {{v[0-9]+}}.h }[{{[0-9]+}}], [x0]
entry:
  %0 = load i16, i16* %a, align 2
  %vld1_lane = insertelement <8 x i16> %b, i16 %0, i32 7
  ret <8 x i16> %vld1_lane
}

define <4 x i32> @test_vld1q_lane_s32(i32* %a, <4 x i32> %b) {
; CHECK-LABEL: test_vld1q_lane_s32:
; CHECK: ld1 { {{v[0-9]+}}.s }[{{[0-9]+}}], [x0]
entry:
  %0 = load i32, i32* %a, align 4
  %vld1_lane = insertelement <4 x i32> %b, i32 %0, i32 3
  ret <4 x i32> %vld1_lane
}

define <2 x i64> @test_vld1q_lane_s64(i64* %a, <2 x i64> %b) {
; CHECK-LABEL: test_vld1q_lane_s64:
; CHECK: ld1 { {{v[0-9]+}}.d }[{{[0-9]+}}], [x0]
entry:
  %0 = load i64, i64* %a, align 8
  %vld1_lane = insertelement <2 x i64> %b, i64 %0, i32 1
  ret <2 x i64> %vld1_lane
}

define <4 x float> @test_vld1q_lane_f32(float* %a, <4 x float> %b) {
; CHECK-LABEL: test_vld1q_lane_f32:
; CHECK: ld1 { {{v[0-9]+}}.s }[{{[0-9]+}}], [x0]
entry:
  %0 = load float, float* %a, align 4
  %vld1_lane = insertelement <4 x float> %b, float %0, i32 3
  ret <4 x float> %vld1_lane
}

define <2 x double> @test_vld1q_lane_f64(double* %a, <2 x double> %b) {
; CHECK-LABEL: test_vld1q_lane_f64:
; CHECK: ld1 { {{v[0-9]+}}.d }[{{[0-9]+}}], [x0]
entry:
  %0 = load double, double* %a, align 8
  %vld1_lane = insertelement <2 x double> %b, double %0, i32 1
  ret <2 x double> %vld1_lane
}

define <8 x i8> @test_vld1_lane_s8(i8* %a, <8 x i8> %b) {
; CHECK-LABEL: test_vld1_lane_s8:
; CHECK: ld1 { {{v[0-9]+}}.b }[{{[0-9]+}}], [x0]
entry:
  %0 = load i8, i8* %a, align 1
  %vld1_lane = insertelement <8 x i8> %b, i8 %0, i32 7
  ret <8 x i8> %vld1_lane
}

define <4 x i16> @test_vld1_lane_s16(i16* %a, <4 x i16> %b) {
; CHECK-LABEL: test_vld1_lane_s16:
; CHECK: ld1 { {{v[0-9]+}}.h }[{{[0-9]+}}], [x0]
entry:
  %0 = load i16, i16* %a, align 2
  %vld1_lane = insertelement <4 x i16> %b, i16 %0, i32 3
  ret <4 x i16> %vld1_lane
}

define <2 x i32> @test_vld1_lane_s32(i32* %a, <2 x i32> %b) {
; CHECK-LABEL: test_vld1_lane_s32:
; CHECK: ld1 { {{v[0-9]+}}.s }[{{[0-9]+}}], [x0]
entry:
  %0 = load i32, i32* %a, align 4
  %vld1_lane = insertelement <2 x i32> %b, i32 %0, i32 1
  ret <2 x i32> %vld1_lane
}

define <1 x i64> @test_vld1_lane_s64(i64* %a, <1 x i64> %b) {
; CHECK-LABEL: test_vld1_lane_s64:
; CHECK: ldr {{d[0-9]+}}, [x0]
entry:
  %0 = load i64, i64* %a, align 8
  %vld1_lane = insertelement <1 x i64> undef, i64 %0, i32 0
  ret <1 x i64> %vld1_lane
}

define <2 x float> @test_vld1_lane_f32(float* %a, <2 x float> %b) {
; CHECK-LABEL: test_vld1_lane_f32:
; CHECK: ld1 { {{v[0-9]+}}.s }[{{[0-9]+}}], [x0]
entry:
  %0 = load float, float* %a, align 4
  %vld1_lane = insertelement <2 x float> %b, float %0, i32 1
  ret <2 x float> %vld1_lane
}

define <1 x double> @test_vld1_lane_f64(double* %a, <1 x double> %b) {
; CHECK-LABEL: test_vld1_lane_f64:
; CHECK: ldr {{d[0-9]+}}, [x0]
entry:
  %0 = load double, double* %a, align 8
  %vld1_lane = insertelement <1 x double> undef, double %0, i32 0
  ret <1 x double> %vld1_lane
}

define void @test_vst1q_lane_s8(i8* %a, <16 x i8> %b) {
; CHECK-LABEL: test_vst1q_lane_s8:
; CHECK: st1 { {{v[0-9]+}}.b }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <16 x i8> %b, i32 15
  store i8 %0, i8* %a, align 1
  ret void
}

define void @test_vst1q_lane_s16(i16* %a, <8 x i16> %b) {
; CHECK-LABEL: test_vst1q_lane_s16:
; CHECK: st1 { {{v[0-9]+}}.h }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <8 x i16> %b, i32 7
  store i16 %0, i16* %a, align 2
  ret void
}

define void @test_vst1q_lane0_s16(i16* %a, <8 x i16> %b) {
; CHECK-LABEL: test_vst1q_lane0_s16:
; CHECK: str {{h[0-9]+}}, [x0]
entry:
  %0 = extractelement <8 x i16> %b, i32 0
  store i16 %0, i16* %a, align 2
  ret void
}

define void @test_vst1q_lane_s32(i32* %a, <4 x i32> %b) {
; CHECK-LABEL: test_vst1q_lane_s32:
; CHECK: st1 { {{v[0-9]+}}.s }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <4 x i32> %b, i32 3
  store i32 %0, i32* %a, align 4
  ret void
}

define void @test_vst1q_lane0_s32(i32* %a, <4 x i32> %b) {
; CHECK-LABEL: test_vst1q_lane0_s32:
; CHECK: str {{s[0-9]+}}, [x0]
entry:
  %0 = extractelement <4 x i32> %b, i32 0
  store i32 %0, i32* %a, align 4
  ret void
}

define void @test_vst1q_lane_s64(i64* %a, <2 x i64> %b) {
; CHECK-LABEL: test_vst1q_lane_s64:
; CHECK: st1 { {{v[0-9]+}}.d }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <2 x i64> %b, i32 1
  store i64 %0, i64* %a, align 8
  ret void
}

define void @test_vst1q_lane0_s64(i64* %a, <2 x i64> %b) {
; CHECK-LABEL: test_vst1q_lane0_s64:
; CHECK: str {{d[0-9]+}}, [x0]
entry:
  %0 = extractelement <2 x i64> %b, i32 0
  store i64 %0, i64* %a, align 8
  ret void
}

define void @test_vst1q_lane_f32(float* %a, <4 x float> %b) {
; CHECK-LABEL: test_vst1q_lane_f32:
; CHECK: st1 { {{v[0-9]+}}.s }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <4 x float> %b, i32 3
  store float %0, float* %a, align 4
  ret void
}

define void @test_vst1q_lane0_f32(float* %a, <4 x float> %b) {
; CHECK-LABEL: test_vst1q_lane0_f32:
; CHECK: str {{s[0-9]+}}, [x0]
entry:
  %0 = extractelement <4 x float> %b, i32 0
  store float %0, float* %a, align 4
  ret void
}

define void @test_vst1q_lane_f64(double* %a, <2 x double> %b) {
; CHECK-LABEL: test_vst1q_lane_f64:
; CHECK: st1 { {{v[0-9]+}}.d }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <2 x double> %b, i32 1
  store double %0, double* %a, align 8
  ret void
}

define void @test_vst1q_lane0_f64(double* %a, <2 x double> %b) {
; CHECK-LABEL: test_vst1q_lane0_f64:
; CHECK: str {{d[0-9]+}}, [x0]
entry:
  %0 = extractelement <2 x double> %b, i32 0
  store double %0, double* %a, align 8
  ret void
}

define void @test_vst1_lane_s8(i8* %a, <8 x i8> %b) {
; CHECK-LABEL: test_vst1_lane_s8:
; CHECK: st1 { {{v[0-9]+}}.b }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <8 x i8> %b, i32 7
  store i8 %0, i8* %a, align 1
  ret void
}

define void @test_vst1_lane_s16(i16* %a, <4 x i16> %b) {
; CHECK-LABEL: test_vst1_lane_s16:
; CHECK: st1 { {{v[0-9]+}}.h }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <4 x i16> %b, i32 3
  store i16 %0, i16* %a, align 2
  ret void
}

define void @test_vst1_lane0_s16(i16* %a, <4 x i16> %b) {
; CHECK-LABEL: test_vst1_lane0_s16:
; CHECK: str {{h[0-9]+}}, [x0]
entry:
  %0 = extractelement <4 x i16> %b, i32 0
  store i16 %0, i16* %a, align 2
  ret void
}

define void @test_vst1_lane_s32(i32* %a, <2 x i32> %b) {
; CHECK-LABEL: test_vst1_lane_s32:
; CHECK: st1 { {{v[0-9]+}}.s }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <2 x i32> %b, i32 1
  store i32 %0, i32* %a, align 4
  ret void
}

define void @test_vst1_lane0_s32(i32* %a, <2 x i32> %b) {
; CHECK-LABEL: test_vst1_lane0_s32:
; CHECK: str {{s[0-9]+}}, [x0]
entry:
  %0 = extractelement <2 x i32> %b, i32 0
  store i32 %0, i32* %a, align 4
  ret void
}

define void @test_vst1_lane_s64(i64* %a, <1 x i64> %b) {
; CHECK-LABEL: test_vst1_lane_s64:
; CHECK: str {{d[0-9]+}}, [x0]
entry:
  %0 = extractelement <1 x i64> %b, i32 0
  store i64 %0, i64* %a, align 8
  ret void
}

define void @test_vst1_lane_f32(float* %a, <2 x float> %b) {
; CHECK-LABEL: test_vst1_lane_f32:
; CHECK: st1 { {{v[0-9]+}}.s }[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <2 x float> %b, i32 1
  store float %0, float* %a, align 4
  ret void
}

define void @test_vst1_lane0_f32(float* %a, <2 x float> %b) {
; CHECK-LABEL: test_vst1_lane0_f32:
; CHECK: str {{s[0-9]+}}, [x0]
entry:
  %0 = extractelement <2 x float> %b, i32 0
  store float %0, float* %a, align 4
  ret void
}

define void @test_vst1_lane_f64(double* %a, <1 x double> %b) {
; CHECK-LABEL: test_vst1_lane_f64:
; CHECK: str {{d[0-9]+}}, [x0]
entry:
  %0 = extractelement <1 x double> %b, i32 0
  store double %0, double* %a, align 8
  ret void
}
