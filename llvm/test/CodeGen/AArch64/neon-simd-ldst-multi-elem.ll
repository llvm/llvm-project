; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define void @test_ldst1_v16i8(<16 x i8>* %ptr, <16 x i8>* %ptr2) {
; CHECK-LABEL: test_ldst1_v16i8:
; CHECK: ld1 {v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %tmp = load <16 x i8>* %ptr
  store <16 x i8> %tmp, <16 x i8>* %ptr2
  ret void
}

define void @test_ldst1_v8i16(<8 x i16>* %ptr, <8 x i16>* %ptr2) {
; CHECK-LABEL: test_ldst1_v8i16:
; CHECK: ld1 {v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %tmp = load <8 x i16>* %ptr
  store <8 x i16> %tmp, <8 x i16>* %ptr2
  ret void
}

define void @test_ldst1_v4i32(<4 x i32>* %ptr, <4 x i32>* %ptr2) {
; CHECK-LABEL: test_ldst1_v4i32:
; CHECK: ld1 {v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %tmp = load <4 x i32>* %ptr
  store <4 x i32> %tmp, <4 x i32>* %ptr2
  ret void
}

define void @test_ldst1_v2i64(<2 x i64>* %ptr, <2 x i64>* %ptr2) {
; CHECK-LABEL: test_ldst1_v2i64:
; CHECK: ld1 {v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %tmp = load <2 x i64>* %ptr
  store <2 x i64> %tmp, <2 x i64>* %ptr2
  ret void
}

define void @test_ldst1_v8i8(<8 x i8>* %ptr, <8 x i8>* %ptr2) {
; CHECK-LABEL: test_ldst1_v8i8:
; CHECK: ld1 {v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %tmp = load <8 x i8>* %ptr
  store <8 x i8> %tmp, <8 x i8>* %ptr2
  ret void
}

define void @test_ldst1_v4i16(<4 x i16>* %ptr, <4 x i16>* %ptr2) {
; CHECK-LABEL: test_ldst1_v4i16:
; CHECK: ld1 {v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %tmp = load <4 x i16>* %ptr
  store <4 x i16> %tmp, <4 x i16>* %ptr2
  ret void
}

define void @test_ldst1_v2i32(<2 x i32>* %ptr, <2 x i32>* %ptr2) {
; CHECK-LABEL: test_ldst1_v2i32:
; CHECK: ld1 {v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %tmp = load <2 x i32>* %ptr
  store <2 x i32> %tmp, <2 x i32>* %ptr2
  ret void
}

define void @test_ldst1_v1i64(<1 x i64>* %ptr, <1 x i64>* %ptr2) {
; CHECK-LABEL: test_ldst1_v1i64:
; CHECK: ld1 {v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
; CHECK: st1 {v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %tmp = load <1 x i64>* %ptr
  store <1 x i64> %tmp, <1 x i64>* %ptr2
  ret void
}

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


define <16 x i8> @test_vld1q_s8(i8* readonly %a) {
; CHECK-LABEL: test_vld1q_s8
; CHECK: ld1 {v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %vld1 = tail call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %a, i32 1)
  ret <16 x i8> %vld1
}

define <8 x i16> @test_vld1q_s16(i16* readonly %a) {
; CHECK-LABEL: test_vld1q_s16
; CHECK: ld1 {v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld1 = tail call <8 x i16> @llvm.arm.neon.vld1.v8i16(i8* %1, i32 2)
  ret <8 x i16> %vld1
}

define <4 x i32> @test_vld1q_s32(i32* readonly %a) {
; CHECK-LABEL: test_vld1q_s32
; CHECK: ld1 {v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld1 = tail call <4 x i32> @llvm.arm.neon.vld1.v4i32(i8* %1, i32 4)
  ret <4 x i32> %vld1
}

define <2 x i64> @test_vld1q_s64(i64* readonly %a) {
; CHECK-LABEL: test_vld1q_s64
; CHECK: ld1 {v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld1 = tail call <2 x i64> @llvm.arm.neon.vld1.v2i64(i8* %1, i32 8)
  ret <2 x i64> %vld1
}

define <4 x float> @test_vld1q_f32(float* readonly %a) {
; CHECK-LABEL: test_vld1q_f32
; CHECK: ld1 {v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld1 = tail call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* %1, i32 4)
  ret <4 x float> %vld1
}

define <2 x double> @test_vld1q_f64(double* readonly %a) {
; CHECK-LABEL: test_vld1q_f64
; CHECK: ld1 {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld1 = tail call <2 x double> @llvm.arm.neon.vld1.v2f64(i8* %1, i32 8)
  ret <2 x double> %vld1
}

define <8 x i8> @test_vld1_s8(i8* readonly %a) {
; CHECK-LABEL: test_vld1_s8
; CHECK: ld1 {v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld1 = tail call <8 x i8> @llvm.arm.neon.vld1.v8i8(i8* %a, i32 1)
  ret <8 x i8> %vld1
}

define <4 x i16> @test_vld1_s16(i16* readonly %a) {
; CHECK-LABEL: test_vld1_s16
; CHECK: ld1 {v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld1 = tail call <4 x i16> @llvm.arm.neon.vld1.v4i16(i8* %1, i32 2)
  ret <4 x i16> %vld1
}

define <2 x i32> @test_vld1_s32(i32* readonly %a) {
; CHECK-LABEL: test_vld1_s32
; CHECK: ld1 {v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld1 = tail call <2 x i32> @llvm.arm.neon.vld1.v2i32(i8* %1, i32 4)
  ret <2 x i32> %vld1
}

define <1 x i64> @test_vld1_s64(i64* readonly %a) {
; CHECK-LABEL: test_vld1_s64
; CHECK: ld1 {v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld1 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %1, i32 8)
  ret <1 x i64> %vld1
}

define <2 x float> @test_vld1_f32(float* readonly %a) {
; CHECK-LABEL: test_vld1_f32
; CHECK: ld1 {v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld1 = tail call <2 x float> @llvm.arm.neon.vld1.v2f32(i8* %1, i32 4)
  ret <2 x float> %vld1
}

define <1 x double> @test_vld1_f64(double* readonly %a) {
; CHECK-LABEL: test_vld1_f64
; CHECK: ld1 {v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld1 = tail call <1 x double> @llvm.arm.neon.vld1.v1f64(i8* %1, i32 8)
  ret <1 x double> %vld1
}

define <8 x i8> @test_vld1_p8(i8* readonly %a) {
; CHECK-LABEL: test_vld1_p8
; CHECK: ld1 {v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld1 = tail call <8 x i8> @llvm.arm.neon.vld1.v8i8(i8* %a, i32 1)
  ret <8 x i8> %vld1
}

define <4 x i16> @test_vld1_p16(i16* readonly %a) {
; CHECK-LABEL: test_vld1_p16
; CHECK: ld1 {v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld1 = tail call <4 x i16> @llvm.arm.neon.vld1.v4i16(i8* %1, i32 2)
  ret <4 x i16> %vld1
}

define %struct.int8x16x2_t @test_vld2q_s8(i8* readonly %a) {
; CHECK-LABEL: test_vld2q_s8
; CHECK: ld2 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %vld2 = tail call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2.v16i8(i8* %a, i32 1)
  %vld2.fca.0.extract = extractvalue { <16 x i8>, <16 x i8> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <16 x i8>, <16 x i8> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int8x16x2_t undef, <16 x i8> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x2_t %.fca.0.0.insert, <16 x i8> %vld2.fca.1.extract, 0, 1
  ret %struct.int8x16x2_t %.fca.0.1.insert
}

define %struct.int16x8x2_t @test_vld2q_s16(i16* readonly %a) {
; CHECK-LABEL: test_vld2q_s16
; CHECK: ld2 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld2 = tail call { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2.v8i16(i8* %1, i32 2)
  %vld2.fca.0.extract = extractvalue { <8 x i16>, <8 x i16> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <8 x i16>, <8 x i16> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int16x8x2_t undef, <8 x i16> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x2_t %.fca.0.0.insert, <8 x i16> %vld2.fca.1.extract, 0, 1
  ret %struct.int16x8x2_t %.fca.0.1.insert
}

define %struct.int32x4x2_t @test_vld2q_s32(i32* readonly %a) {
; CHECK-LABEL: test_vld2q_s32
; CHECK: ld2 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld2 = tail call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32(i8* %1, i32 4)
  %vld2.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int32x4x2_t undef, <4 x i32> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x2_t %.fca.0.0.insert, <4 x i32> %vld2.fca.1.extract, 0, 1
  ret %struct.int32x4x2_t %.fca.0.1.insert
}

define %struct.int64x2x2_t @test_vld2q_s64(i64* readonly %a) {
; CHECK-LABEL: test_vld2q_s64
; CHECK: ld2 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld2 = tail call { <2 x i64>, <2 x i64> } @llvm.arm.neon.vld2.v2i64(i8* %1, i32 8)
  %vld2.fca.0.extract = extractvalue { <2 x i64>, <2 x i64> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <2 x i64>, <2 x i64> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int64x2x2_t undef, <2 x i64> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x2_t %.fca.0.0.insert, <2 x i64> %vld2.fca.1.extract, 0, 1
  ret %struct.int64x2x2_t %.fca.0.1.insert
}

define %struct.float32x4x2_t @test_vld2q_f32(float* readonly %a) {
; CHECK-LABEL: test_vld2q_f32
; CHECK: ld2 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld2 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8* %1, i32 4)
  %vld2.fca.0.extract = extractvalue { <4 x float>, <4 x float> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x float>, <4 x float> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.float32x4x2_t undef, <4 x float> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x2_t %.fca.0.0.insert, <4 x float> %vld2.fca.1.extract, 0, 1
  ret %struct.float32x4x2_t %.fca.0.1.insert
}

define %struct.float64x2x2_t @test_vld2q_f64(double* readonly %a) {
; CHECK-LABEL: test_vld2q_f64
; CHECK: ld2 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld2 = tail call { <2 x double>, <2 x double> } @llvm.arm.neon.vld2.v2f64(i8* %1, i32 8)
  %vld2.fca.0.extract = extractvalue { <2 x double>, <2 x double> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <2 x double>, <2 x double> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.float64x2x2_t undef, <2 x double> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x2_t %.fca.0.0.insert, <2 x double> %vld2.fca.1.extract, 0, 1
  ret %struct.float64x2x2_t %.fca.0.1.insert
}

define %struct.int8x8x2_t @test_vld2_s8(i8* readonly %a) {
; CHECK-LABEL: test_vld2_s8
; CHECK: ld2 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld2 = tail call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2.v8i8(i8* %a, i32 1)
  %vld2.fca.0.extract = extractvalue { <8 x i8>, <8 x i8> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <8 x i8>, <8 x i8> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int8x8x2_t undef, <8 x i8> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x2_t %.fca.0.0.insert, <8 x i8> %vld2.fca.1.extract, 0, 1
  ret %struct.int8x8x2_t %.fca.0.1.insert
}

define %struct.int16x4x2_t @test_vld2_s16(i16* readonly %a) {
; CHECK-LABEL: test_vld2_s16
; CHECK: ld2 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld2 = tail call { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2.v4i16(i8* %1, i32 2)
  %vld2.fca.0.extract = extractvalue { <4 x i16>, <4 x i16> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x i16>, <4 x i16> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int16x4x2_t undef, <4 x i16> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x2_t %.fca.0.0.insert, <4 x i16> %vld2.fca.1.extract, 0, 1
  ret %struct.int16x4x2_t %.fca.0.1.insert
}

define %struct.int32x2x2_t @test_vld2_s32(i32* readonly %a) {
; CHECK-LABEL: test_vld2_s32
; CHECK: ld2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld2 = tail call { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2.v2i32(i8* %1, i32 4)
  %vld2.fca.0.extract = extractvalue { <2 x i32>, <2 x i32> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <2 x i32>, <2 x i32> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int32x2x2_t undef, <2 x i32> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x2_t %.fca.0.0.insert, <2 x i32> %vld2.fca.1.extract, 0, 1
  ret %struct.int32x2x2_t %.fca.0.1.insert
}

define %struct.int64x1x2_t @test_vld2_s64(i64* readonly %a) {
; CHECK-LABEL: test_vld2_s64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld2 = tail call { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2.v1i64(i8* %1, i32 8)
  %vld2.fca.0.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int64x1x2_t undef, <1 x i64> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x2_t %.fca.0.0.insert, <1 x i64> %vld2.fca.1.extract, 0, 1
  ret %struct.int64x1x2_t %.fca.0.1.insert
}

define %struct.float32x2x2_t @test_vld2_f32(float* readonly %a) {
; CHECK-LABEL: test_vld2_f32
; CHECK: ld2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld2 = tail call { <2 x float>, <2 x float> } @llvm.arm.neon.vld2.v2f32(i8* %1, i32 4)
  %vld2.fca.0.extract = extractvalue { <2 x float>, <2 x float> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <2 x float>, <2 x float> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.float32x2x2_t undef, <2 x float> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x2_t %.fca.0.0.insert, <2 x float> %vld2.fca.1.extract, 0, 1
  ret %struct.float32x2x2_t %.fca.0.1.insert
}

define %struct.float64x1x2_t @test_vld2_f64(double* readonly %a) {
; CHECK-LABEL: test_vld2_f64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld2 = tail call { <1 x double>, <1 x double> } @llvm.arm.neon.vld2.v1f64(i8* %1, i32 8)
  %vld2.fca.0.extract = extractvalue { <1 x double>, <1 x double> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <1 x double>, <1 x double> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.float64x1x2_t undef, <1 x double> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x2_t %.fca.0.0.insert, <1 x double> %vld2.fca.1.extract, 0, 1
  ret %struct.float64x1x2_t %.fca.0.1.insert
}

define %struct.int8x16x3_t @test_vld3q_s8(i8* readonly %a) {
; CHECK-LABEL: test_vld3q_s8
; CHECK: ld3 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %vld3 = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8(i8* %a, i32 1)
  %vld3.fca.0.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int8x16x3_t undef, <16 x i8> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x3_t %.fca.0.0.insert, <16 x i8> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x16x3_t %.fca.0.1.insert, <16 x i8> %vld3.fca.2.extract, 0, 2
  ret %struct.int8x16x3_t %.fca.0.2.insert
}

define %struct.int16x8x3_t @test_vld3q_s16(i16* readonly %a) {
; CHECK-LABEL: test_vld3q_s16
; CHECK: ld3 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld3 = tail call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3.v8i16(i8* %1, i32 2)
  %vld3.fca.0.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int16x8x3_t undef, <8 x i16> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x3_t %.fca.0.0.insert, <8 x i16> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x8x3_t %.fca.0.1.insert, <8 x i16> %vld3.fca.2.extract, 0, 2
  ret %struct.int16x8x3_t %.fca.0.2.insert
}

define %struct.int32x4x3_t @test_vld3q_s32(i32* readonly %a) {
; CHECK-LABEL: test_vld3q_s32
; CHECK: ld3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld3 = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32(i8* %1, i32 4)
  %vld3.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int32x4x3_t undef, <4 x i32> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x3_t %.fca.0.0.insert, <4 x i32> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x4x3_t %.fca.0.1.insert, <4 x i32> %vld3.fca.2.extract, 0, 2
  ret %struct.int32x4x3_t %.fca.0.2.insert
}

define %struct.int64x2x3_t @test_vld3q_s64(i64* readonly %a) {
; CHECK-LABEL: test_vld3q_s64
; CHECK: ld3 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld3 = tail call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld3.v2i64(i8* %1, i32 8)
  %vld3.fca.0.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int64x2x3_t undef, <2 x i64> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x3_t %.fca.0.0.insert, <2 x i64> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x2x3_t %.fca.0.1.insert, <2 x i64> %vld3.fca.2.extract, 0, 2
  ret %struct.int64x2x3_t %.fca.0.2.insert
}

define %struct.float32x4x3_t @test_vld3q_f32(float* readonly %a) {
; CHECK-LABEL: test_vld3q_f32
; CHECK: ld3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld3 = tail call { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3.v4f32(i8* %1, i32 4)
  %vld3.fca.0.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.float32x4x3_t undef, <4 x float> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x3_t %.fca.0.0.insert, <4 x float> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x4x3_t %.fca.0.1.insert, <4 x float> %vld3.fca.2.extract, 0, 2
  ret %struct.float32x4x3_t %.fca.0.2.insert
}

define %struct.float64x2x3_t @test_vld3q_f64(double* readonly %a) {
; CHECK-LABEL: test_vld3q_f64
; CHECK: ld3 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld3 = tail call { <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld3.v2f64(i8* %1, i32 8)
  %vld3.fca.0.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.float64x2x3_t undef, <2 x double> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x3_t %.fca.0.0.insert, <2 x double> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x2x3_t %.fca.0.1.insert, <2 x double> %vld3.fca.2.extract, 0, 2
  ret %struct.float64x2x3_t %.fca.0.2.insert
}

define %struct.int8x8x3_t @test_vld3_s8(i8* readonly %a) {
; CHECK-LABEL: test_vld3_s8
; CHECK: ld3 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld3 = tail call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3.v8i8(i8* %a, i32 1)
  %vld3.fca.0.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int8x8x3_t undef, <8 x i8> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x3_t %.fca.0.0.insert, <8 x i8> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x8x3_t %.fca.0.1.insert, <8 x i8> %vld3.fca.2.extract, 0, 2
  ret %struct.int8x8x3_t %.fca.0.2.insert
}

define %struct.int16x4x3_t @test_vld3_s16(i16* readonly %a) {
; CHECK-LABEL: test_vld3_s16
; CHECK: ld3 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld3 = tail call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3.v4i16(i8* %1, i32 2)
  %vld3.fca.0.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int16x4x3_t undef, <4 x i16> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x3_t %.fca.0.0.insert, <4 x i16> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x4x3_t %.fca.0.1.insert, <4 x i16> %vld3.fca.2.extract, 0, 2
  ret %struct.int16x4x3_t %.fca.0.2.insert
}

define %struct.int32x2x3_t @test_vld3_s32(i32* readonly %a) {
; CHECK-LABEL: test_vld3_s32
; CHECK: ld3 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld3 = tail call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3.v2i32(i8* %1, i32 4)
  %vld3.fca.0.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int32x2x3_t undef, <2 x i32> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x3_t %.fca.0.0.insert, <2 x i32> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x2x3_t %.fca.0.1.insert, <2 x i32> %vld3.fca.2.extract, 0, 2
  ret %struct.int32x2x3_t %.fca.0.2.insert
}

define %struct.int64x1x3_t @test_vld3_s64(i64* readonly %a) {
; CHECK-LABEL: test_vld3_s64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld3 = tail call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3.v1i64(i8* %1, i32 8)
  %vld3.fca.0.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int64x1x3_t undef, <1 x i64> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x3_t %.fca.0.0.insert, <1 x i64> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x1x3_t %.fca.0.1.insert, <1 x i64> %vld3.fca.2.extract, 0, 2
  ret %struct.int64x1x3_t %.fca.0.2.insert
}

define %struct.float32x2x3_t @test_vld3_f32(float* readonly %a) {
; CHECK-LABEL: test_vld3_f32
; CHECK: ld3 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld3 = tail call { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3.v2f32(i8* %1, i32 4)
  %vld3.fca.0.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.float32x2x3_t undef, <2 x float> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x3_t %.fca.0.0.insert, <2 x float> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x2x3_t %.fca.0.1.insert, <2 x float> %vld3.fca.2.extract, 0, 2
  ret %struct.float32x2x3_t %.fca.0.2.insert
}

define %struct.float64x1x3_t @test_vld3_f64(double* readonly %a) {
; CHECK-LABEL: test_vld3_f64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld3 = tail call { <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld3.v1f64(i8* %1, i32 8)
  %vld3.fca.0.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.float64x1x3_t undef, <1 x double> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x3_t %.fca.0.0.insert, <1 x double> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x1x3_t %.fca.0.1.insert, <1 x double> %vld3.fca.2.extract, 0, 2
  ret %struct.float64x1x3_t %.fca.0.2.insert
}

define %struct.int8x16x4_t @test_vld4q_s8(i8* readonly %a) {
; CHECK-LABEL: test_vld4q_s8
; CHECK: ld4 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %vld4 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4.v16i8(i8* %a, i32 1)
  %vld4.fca.0.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int8x16x4_t undef, <16 x i8> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x4_t %.fca.0.0.insert, <16 x i8> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x16x4_t %.fca.0.1.insert, <16 x i8> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int8x16x4_t %.fca.0.2.insert, <16 x i8> %vld4.fca.3.extract, 0, 3
  ret %struct.int8x16x4_t %.fca.0.3.insert
}

define %struct.int16x8x4_t @test_vld4q_s16(i16* readonly %a) {
; CHECK-LABEL: test_vld4q_s16
; CHECK: ld4 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld4 = tail call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4.v8i16(i8* %1, i32 2)
  %vld4.fca.0.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int16x8x4_t undef, <8 x i16> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x4_t %.fca.0.0.insert, <8 x i16> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x8x4_t %.fca.0.1.insert, <8 x i16> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int16x8x4_t %.fca.0.2.insert, <8 x i16> %vld4.fca.3.extract, 0, 3
  ret %struct.int16x8x4_t %.fca.0.3.insert
}

define %struct.int32x4x4_t @test_vld4q_s32(i32* readonly %a) {
; CHECK-LABEL: test_vld4q_s32
; CHECK: ld4 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld4 = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4.v4i32(i8* %1, i32 4)
  %vld4.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int32x4x4_t undef, <4 x i32> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x4_t %.fca.0.0.insert, <4 x i32> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x4x4_t %.fca.0.1.insert, <4 x i32> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int32x4x4_t %.fca.0.2.insert, <4 x i32> %vld4.fca.3.extract, 0, 3
  ret %struct.int32x4x4_t %.fca.0.3.insert
}

define %struct.int64x2x4_t @test_vld4q_s64(i64* readonly %a) {
; CHECK-LABEL: test_vld4q_s64
; CHECK: ld4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld4 = tail call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld4.v2i64(i8* %1, i32 8)
  %vld4.fca.0.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int64x2x4_t undef, <2 x i64> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x4_t %.fca.0.0.insert, <2 x i64> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x2x4_t %.fca.0.1.insert, <2 x i64> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int64x2x4_t %.fca.0.2.insert, <2 x i64> %vld4.fca.3.extract, 0, 3
  ret %struct.int64x2x4_t %.fca.0.3.insert
}

define %struct.float32x4x4_t @test_vld4q_f32(float* readonly %a) {
; CHECK-LABEL: test_vld4q_f32
; CHECK: ld4 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld4 = tail call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4.v4f32(i8* %1, i32 4)
  %vld4.fca.0.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.float32x4x4_t undef, <4 x float> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x4_t %.fca.0.0.insert, <4 x float> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x4x4_t %.fca.0.1.insert, <4 x float> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float32x4x4_t %.fca.0.2.insert, <4 x float> %vld4.fca.3.extract, 0, 3
  ret %struct.float32x4x4_t %.fca.0.3.insert
}

define %struct.float64x2x4_t @test_vld4q_f64(double* readonly %a) {
; CHECK-LABEL: test_vld4q_f64
; CHECK: ld4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld4 = tail call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4.v2f64(i8* %1, i32 8)
  %vld4.fca.0.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.float64x2x4_t undef, <2 x double> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x4_t %.fca.0.0.insert, <2 x double> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x2x4_t %.fca.0.1.insert, <2 x double> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float64x2x4_t %.fca.0.2.insert, <2 x double> %vld4.fca.3.extract, 0, 3
  ret %struct.float64x2x4_t %.fca.0.3.insert
}

define %struct.int8x8x4_t @test_vld4_s8(i8* readonly %a) {
; CHECK-LABEL: test_vld4_s8
; CHECK: ld4 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld4 = tail call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4.v8i8(i8* %a, i32 1)
  %vld4.fca.0.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int8x8x4_t undef, <8 x i8> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x4_t %.fca.0.0.insert, <8 x i8> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x8x4_t %.fca.0.1.insert, <8 x i8> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int8x8x4_t %.fca.0.2.insert, <8 x i8> %vld4.fca.3.extract, 0, 3
  ret %struct.int8x8x4_t %.fca.0.3.insert
}

define %struct.int16x4x4_t @test_vld4_s16(i16* readonly %a) {
; CHECK-LABEL: test_vld4_s16
; CHECK: ld4 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld4 = tail call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4.v4i16(i8* %1, i32 2)
  %vld4.fca.0.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int16x4x4_t undef, <4 x i16> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x4_t %.fca.0.0.insert, <4 x i16> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x4x4_t %.fca.0.1.insert, <4 x i16> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int16x4x4_t %.fca.0.2.insert, <4 x i16> %vld4.fca.3.extract, 0, 3
  ret %struct.int16x4x4_t %.fca.0.3.insert
}

define %struct.int32x2x4_t @test_vld4_s32(i32* readonly %a) {
; CHECK-LABEL: test_vld4_s32
; CHECK: ld4 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld4 = tail call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4.v2i32(i8* %1, i32 4)
  %vld4.fca.0.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int32x2x4_t undef, <2 x i32> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x4_t %.fca.0.0.insert, <2 x i32> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x2x4_t %.fca.0.1.insert, <2 x i32> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int32x2x4_t %.fca.0.2.insert, <2 x i32> %vld4.fca.3.extract, 0, 3
  ret %struct.int32x2x4_t %.fca.0.3.insert
}

define %struct.int64x1x4_t @test_vld4_s64(i64* readonly %a) {
; CHECK-LABEL: test_vld4_s64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld4 = tail call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4.v1i64(i8* %1, i32 8)
  %vld4.fca.0.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int64x1x4_t undef, <1 x i64> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x4_t %.fca.0.0.insert, <1 x i64> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x1x4_t %.fca.0.1.insert, <1 x i64> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int64x1x4_t %.fca.0.2.insert, <1 x i64> %vld4.fca.3.extract, 0, 3
  ret %struct.int64x1x4_t %.fca.0.3.insert
}

define %struct.float32x2x4_t @test_vld4_f32(float* readonly %a) {
; CHECK-LABEL: test_vld4_f32
; CHECK: ld4 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld4 = tail call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld4.v2f32(i8* %1, i32 4)
  %vld4.fca.0.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.float32x2x4_t undef, <2 x float> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x4_t %.fca.0.0.insert, <2 x float> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x2x4_t %.fca.0.1.insert, <2 x float> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float32x2x4_t %.fca.0.2.insert, <2 x float> %vld4.fca.3.extract, 0, 3
  ret %struct.float32x2x4_t %.fca.0.3.insert
}

define %struct.float64x1x4_t @test_vld4_f64(double* readonly %a) {
; CHECK-LABEL: test_vld4_f64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld4 = tail call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld4.v1f64(i8* %1, i32 8)
  %vld4.fca.0.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.float64x1x4_t undef, <1 x double> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x4_t %.fca.0.0.insert, <1 x double> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x1x4_t %.fca.0.1.insert, <1 x double> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float64x1x4_t %.fca.0.2.insert, <1 x double> %vld4.fca.3.extract, 0, 3
  ret %struct.float64x1x4_t %.fca.0.3.insert
}

declare <16 x i8> @llvm.arm.neon.vld1.v16i8(i8*, i32)
declare <8 x i16> @llvm.arm.neon.vld1.v8i16(i8*, i32)
declare <4 x i32> @llvm.arm.neon.vld1.v4i32(i8*, i32)
declare <2 x i64> @llvm.arm.neon.vld1.v2i64(i8*, i32)
declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*, i32)
declare <2 x double> @llvm.arm.neon.vld1.v2f64(i8*, i32)
declare <8 x i8> @llvm.arm.neon.vld1.v8i8(i8*, i32)
declare <4 x i16> @llvm.arm.neon.vld1.v4i16(i8*, i32)
declare <2 x i32> @llvm.arm.neon.vld1.v2i32(i8*, i32)
declare <1 x i64> @llvm.arm.neon.vld1.v1i64(i8*, i32)
declare <2 x float> @llvm.arm.neon.vld1.v2f32(i8*, i32)
declare <1 x double> @llvm.arm.neon.vld1.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64> } @llvm.arm.neon.vld2.v2i64(i8*, i32)
declare { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8*, i32)
declare { <2 x double>, <2 x double> } @llvm.arm.neon.vld2.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2.v1i64(i8*, i32)
declare { <2 x float>, <2 x float> } @llvm.arm.neon.vld2.v2f32(i8*, i32)
declare { <1 x double>, <1 x double> } @llvm.arm.neon.vld2.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld3.v2i64(i8*, i32)
declare { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3.v4f32(i8*, i32)
declare { <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld3.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3.v1i64(i8*, i32)
declare { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3.v2f32(i8*, i32)
declare { <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld3.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld4.v2i64(i8*, i32)
declare { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4.v4f32(i8*, i32)
declare { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4.v1i64(i8*, i32)
declare { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld4.v2f32(i8*, i32)
declare { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld4.v1f64(i8*, i32)

define void @test_vst1q_s8(i8* %a, <16 x i8> %b) {
; CHECK-LABEL: test_vst1q_s8
; CHECK: st1 {v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  tail call void @llvm.arm.neon.vst1.v16i8(i8* %a, <16 x i8> %b, i32 1)
  ret void
}

define void @test_vst1q_s16(i16* %a, <8 x i16> %b) {
; CHECK-LABEL: test_vst1q_s16
; CHECK: st1 {v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst1.v8i16(i8* %1, <8 x i16> %b, i32 2)
  ret void
}

define void @test_vst1q_s32(i32* %a, <4 x i32> %b) {
; CHECK-LABEL: test_vst1q_s32
; CHECK: st1 {v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst1.v4i32(i8* %1, <4 x i32> %b, i32 4)
  ret void
}

define void @test_vst1q_s64(i64* %a, <2 x i64> %b) {
; CHECK-LABEL: test_vst1q_s64
; CHECK: st1 {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst1.v2i64(i8* %1, <2 x i64> %b, i32 8)
  ret void
}

define void @test_vst1q_f32(float* %a, <4 x float> %b) {
; CHECK-LABEL: test_vst1q_f32
; CHECK: st1 {v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst1.v4f32(i8* %1, <4 x float> %b, i32 4)
  ret void
}

define void @test_vst1q_f64(double* %a, <2 x double> %b) {
; CHECK-LABEL: test_vst1q_f64
; CHECK: st1 {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst1.v2f64(i8* %1, <2 x double> %b, i32 8)
  ret void
}

define void @test_vst1_s8(i8* %a, <8 x i8> %b) {
; CHECK-LABEL: test_vst1_s8
; CHECK: st1 {v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  tail call void @llvm.arm.neon.vst1.v8i8(i8* %a, <8 x i8> %b, i32 1)
  ret void
}

define void @test_vst1_s16(i16* %a, <4 x i16> %b) {
; CHECK-LABEL: test_vst1_s16
; CHECK: st1 {v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst1.v4i16(i8* %1, <4 x i16> %b, i32 2)
  ret void
}

define void @test_vst1_s32(i32* %a, <2 x i32> %b) {
; CHECK-LABEL: test_vst1_s32
; CHECK: st1 {v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst1.v2i32(i8* %1, <2 x i32> %b, i32 4)
  ret void
}

define void @test_vst1_s64(i64* %a, <1 x i64> %b) {
; CHECK-LABEL: test_vst1_s64
; CHECK: st1 {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst1.v1i64(i8* %1, <1 x i64> %b, i32 8)
  ret void
}

define void @test_vst1_f32(float* %a, <2 x float> %b) {
; CHECK-LABEL: test_vst1_f32
; CHECK: st1 {v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst1.v2f32(i8* %1, <2 x float> %b, i32 4)
  ret void
}

define void @test_vst1_f64(double* %a, <1 x double> %b) {
; CHECK-LABEL: test_vst1_f64
; CHECK: st1 {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst1.v1f64(i8* %1, <1 x double> %b, i32 8)
  ret void
}

define void @test_vst2q_s8(i8* %a, [2 x <16 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst2q_s8
; CHECK: st2 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <16 x i8>] %b.coerce, 1
  tail call void @llvm.arm.neon.vst2.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, i32 1)
  ret void
}

define void @test_vst2q_s16(i16* %a, [2 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst2q_s16
; CHECK: st2 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <8 x i16>] %b.coerce, 1
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst2.v8i16(i8* %1, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, i32 2)
  ret void
}

define void @test_vst2q_s32(i32* %a, [2 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst2q_s32
; CHECK: st2 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %b.coerce, 1
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst2.v4i32(i8* %1, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, i32 4)
  ret void
}

define void @test_vst2q_s64(i64* %a, [2 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst2q_s64
; CHECK: st2 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x i64>] %b.coerce, 1
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst2.v2i64(i8* %1, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, i32 8)
  ret void
}

define void @test_vst2q_f32(float* %a, [2 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vst2q_f32
; CHECK: st2 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x float>] %b.coerce, 1
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst2.v4f32(i8* %1, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, i32 4)
  ret void
}

define void @test_vst2q_f64(double* %a, [2 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vst2q_f64
; CHECK: st2 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x double>] %b.coerce, 1
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst2.v2f64(i8* %1, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, i32 8)
  ret void
}

define void @test_vst2_s8(i8* %a, [2 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst2_s8
; CHECK: st2 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <8 x i8>] %b.coerce, 1
  tail call void @llvm.arm.neon.vst2.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, i32 1)
  ret void
}

define void @test_vst2_s16(i16* %a, [2 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst2_s16
; CHECK: st2 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x i16>] %b.coerce, 1
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst2.v4i16(i8* %1, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, i32 2)
  ret void
}

define void @test_vst2_s32(i32* %a, [2 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst2_s32
; CHECK: st2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x i32>] %b.coerce, 1
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst2.v2i32(i8* %1, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, i32 4)
  ret void
}

define void @test_vst2_s64(i64* %a, [2 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst2_s64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <1 x i64>] %b.coerce, 1
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst2.v1i64(i8* %1, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, i32 8)
  ret void
}

define void @test_vst2_f32(float* %a, [2 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vst2_f32
; CHECK: st2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x float>] %b.coerce, 1
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst2.v2f32(i8* %1, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, i32 4)
  ret void
}

define void @test_vst2_f64(double* %a, [2 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vst2_f64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <1 x double>] %b.coerce, 1
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst2.v1f64(i8* %1, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, i32 8)
  ret void
}

define void @test_vst3q_s8(i8* %a, [3 x <16 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst3q_s8
; CHECK: st3 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <16 x i8>] %b.coerce, 2
  tail call void @llvm.arm.neon.vst3.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, <16 x i8> %b.coerce.fca.2.extract, i32 1)
  ret void
}

define void @test_vst3q_s16(i16* %a, [3 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst3q_s16
; CHECK: st3 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <8 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <8 x i16>] %b.coerce, 2
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst3.v8i16(i8* %1, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, <8 x i16> %b.coerce.fca.2.extract, i32 2)
  ret void
}

define void @test_vst3q_s32(i32* %a, [3 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst3q_s32
; CHECK: st3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x i32>] %b.coerce, 2
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst3.v4i32(i8* %1, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, <4 x i32> %b.coerce.fca.2.extract, i32 4)
  ret void
}

define void @test_vst3q_s64(i64* %a, [3 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst3q_s64
; CHECK: st3 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x i64>] %b.coerce, 2
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst3.v2i64(i8* %1, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, <2 x i64> %b.coerce.fca.2.extract, i32 8)
  ret void
}

define void @test_vst3q_f32(float* %a, [3 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vst3q_f32
; CHECK: st3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x float>] %b.coerce, 2
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst3.v4f32(i8* %1, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, <4 x float> %b.coerce.fca.2.extract, i32 4)
  ret void
}

define void @test_vst3q_f64(double* %a, [3 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vst3q_f64
; CHECK: st3 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x double>] %b.coerce, 2
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst3.v2f64(i8* %1, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, <2 x double> %b.coerce.fca.2.extract, i32 8)
  ret void
}

define void @test_vst3_s8(i8* %a, [3 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst3_s8
; CHECK: st3 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <8 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <8 x i8>] %b.coerce, 2
  tail call void @llvm.arm.neon.vst3.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, <8 x i8> %b.coerce.fca.2.extract, i32 1)
  ret void
}

define void @test_vst3_s16(i16* %a, [3 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst3_s16
; CHECK: st3 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x i16>] %b.coerce, 2
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst3.v4i16(i8* %1, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, <4 x i16> %b.coerce.fca.2.extract, i32 2)
  ret void
}

define void @test_vst3_s32(i32* %a, [3 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst3_s32
; CHECK: st3 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x i32>] %b.coerce, 2
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst3.v2i32(i8* %1, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, <2 x i32> %b.coerce.fca.2.extract, i32 4)
  ret void
}

define void @test_vst3_s64(i64* %a, [3 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst3_s64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <1 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <1 x i64>] %b.coerce, 2
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst3.v1i64(i8* %1, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, <1 x i64> %b.coerce.fca.2.extract, i32 8)
  ret void
}

define void @test_vst3_f32(float* %a, [3 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vst3_f32
; CHECK: st3 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x float>] %b.coerce, 2
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst3.v2f32(i8* %1, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, <2 x float> %b.coerce.fca.2.extract, i32 4)
  ret void
}

define void @test_vst3_f64(double* %a, [3 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vst3_f64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <1 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <1 x double>] %b.coerce, 2
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst3.v1f64(i8* %1, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, <1 x double> %b.coerce.fca.2.extract, i32 8)
  ret void
}

define void @test_vst4q_s8(i8* %a, [4 x <16 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst4q_s8
; CHECK: st4 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <16 x i8>] %b.coerce, 3
  tail call void @llvm.arm.neon.vst4.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, <16 x i8> %b.coerce.fca.2.extract, <16 x i8> %b.coerce.fca.3.extract, i32 1)
  ret void
}

define void @test_vst4q_s16(i16* %a, [4 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst4q_s16
; CHECK: st4 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <8 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <8 x i16>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <8 x i16>] %b.coerce, 3
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst4.v8i16(i8* %1, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, <8 x i16> %b.coerce.fca.2.extract, <8 x i16> %b.coerce.fca.3.extract, i32 2)
  ret void
}

define void @test_vst4q_s32(i32* %a, [4 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst4q_s32
; CHECK: st4 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x i32>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x i32>] %b.coerce, 3
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst4.v4i32(i8* %1, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, <4 x i32> %b.coerce.fca.2.extract, <4 x i32> %b.coerce.fca.3.extract, i32 4)
  ret void
}

define void @test_vst4q_s64(i64* %a, [4 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst4q_s64
; CHECK: st4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x i64>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x i64>] %b.coerce, 3
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst4.v2i64(i8* %1, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, <2 x i64> %b.coerce.fca.2.extract, <2 x i64> %b.coerce.fca.3.extract, i32 8)
  ret void
}

define void @test_vst4q_f32(float* %a, [4 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vst4q_f32
; CHECK: st4 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x float>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x float>] %b.coerce, 3
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst4.v4f32(i8* %1, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, <4 x float> %b.coerce.fca.2.extract, <4 x float> %b.coerce.fca.3.extract, i32 4)
  ret void
}

define void @test_vst4q_f64(double* %a, [4 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vst4q_f64
; CHECK: st4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x double>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x double>] %b.coerce, 3
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst4.v2f64(i8* %1, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, <2 x double> %b.coerce.fca.2.extract, <2 x double> %b.coerce.fca.3.extract, i32 8)
  ret void
}

define void @test_vst4_s8(i8* %a, [4 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst4_s8
; CHECK: st4 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <8 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <8 x i8>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <8 x i8>] %b.coerce, 3
  tail call void @llvm.arm.neon.vst4.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, <8 x i8> %b.coerce.fca.2.extract, <8 x i8> %b.coerce.fca.3.extract, i32 1)
  ret void
}

define void @test_vst4_s16(i16* %a, [4 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst4_s16
; CHECK: st4 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x i16>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x i16>] %b.coerce, 3
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst4.v4i16(i8* %1, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, <4 x i16> %b.coerce.fca.2.extract, <4 x i16> %b.coerce.fca.3.extract, i32 2)
  ret void
}

define void @test_vst4_s32(i32* %a, [4 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst4_s32
; CHECK: st4 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x i32>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x i32>] %b.coerce, 3
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst4.v2i32(i8* %1, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, <2 x i32> %b.coerce.fca.2.extract, <2 x i32> %b.coerce.fca.3.extract, i32 4)
  ret void
}

define void @test_vst4_s64(i64* %a, [4 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst4_s64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <1 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <1 x i64>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <1 x i64>] %b.coerce, 3
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst4.v1i64(i8* %1, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, <1 x i64> %b.coerce.fca.2.extract, <1 x i64> %b.coerce.fca.3.extract, i32 8)
  ret void
}

define void @test_vst4_f32(float* %a, [4 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vst4_f32
; CHECK: st4 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x float>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x float>] %b.coerce, 3
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst4.v2f32(i8* %1, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, <2 x float> %b.coerce.fca.2.extract, <2 x float> %b.coerce.fca.3.extract, i32 4)
  ret void
}

define void @test_vst4_f64(double* %a, [4 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vst4_f64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <1 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <1 x double>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <1 x double>] %b.coerce, 3
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst4.v1f64(i8* %1, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, <1 x double> %b.coerce.fca.2.extract, <1 x double> %b.coerce.fca.3.extract, i32 8)
  ret void
}

declare void @llvm.arm.neon.vst1.v16i8(i8*, <16 x i8>, i32)
declare void @llvm.arm.neon.vst1.v8i16(i8*, <8 x i16>, i32)
declare void @llvm.arm.neon.vst1.v4i32(i8*, <4 x i32>, i32)
declare void @llvm.arm.neon.vst1.v2i64(i8*, <2 x i64>, i32)
declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32)
declare void @llvm.arm.neon.vst1.v2f64(i8*, <2 x double>, i32)
declare void @llvm.arm.neon.vst1.v8i8(i8*, <8 x i8>, i32)
declare void @llvm.arm.neon.vst1.v4i16(i8*, <4 x i16>, i32)
declare void @llvm.arm.neon.vst1.v2i32(i8*, <2 x i32>, i32)
declare void @llvm.arm.neon.vst1.v1i64(i8*, <1 x i64>, i32)
declare void @llvm.arm.neon.vst1.v2f32(i8*, <2 x float>, i32)
declare void @llvm.arm.neon.vst1.v1f64(i8*, <1 x double>, i32)
declare void @llvm.arm.neon.vst2.v16i8(i8*, <16 x i8>, <16 x i8>, i32)
declare void @llvm.arm.neon.vst2.v8i16(i8*, <8 x i16>, <8 x i16>, i32)
declare void @llvm.arm.neon.vst2.v4i32(i8*, <4 x i32>, <4 x i32>, i32)
declare void @llvm.arm.neon.vst2.v2i64(i8*, <2 x i64>, <2 x i64>, i32)
declare void @llvm.arm.neon.vst2.v4f32(i8*, <4 x float>, <4 x float>, i32)
declare void @llvm.arm.neon.vst2.v2f64(i8*, <2 x double>, <2 x double>, i32)
declare void @llvm.arm.neon.vst2.v8i8(i8*, <8 x i8>, <8 x i8>, i32)
declare void @llvm.arm.neon.vst2.v4i16(i8*, <4 x i16>, <4 x i16>, i32)
declare void @llvm.arm.neon.vst2.v2i32(i8*, <2 x i32>, <2 x i32>, i32)
declare void @llvm.arm.neon.vst2.v1i64(i8*, <1 x i64>, <1 x i64>, i32)
declare void @llvm.arm.neon.vst2.v2f32(i8*, <2 x float>, <2 x float>, i32)
declare void @llvm.arm.neon.vst2.v1f64(i8*, <1 x double>, <1 x double>, i32)
declare void @llvm.arm.neon.vst3.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, i32)
declare void @llvm.arm.neon.vst3.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32)
declare void @llvm.arm.neon.vst3.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32)
declare void @llvm.arm.neon.vst3.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, i32)
declare void @llvm.arm.neon.vst3.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, i32)
declare void @llvm.arm.neon.vst3.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, i32)
declare void @llvm.arm.neon.vst3.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32)
declare void @llvm.arm.neon.vst3.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32)
declare void @llvm.arm.neon.vst3.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32)
declare void @llvm.arm.neon.vst3.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, i32)
declare void @llvm.arm.neon.vst3.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32)
declare void @llvm.arm.neon.vst3.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, i32)
declare void @llvm.arm.neon.vst4.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32)
declare void @llvm.arm.neon.vst4.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32)
declare void @llvm.arm.neon.vst4.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32)
declare void @llvm.arm.neon.vst4.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, i32)
declare void @llvm.arm.neon.vst4.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32)
declare void @llvm.arm.neon.vst4.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, <2 x double>, i32)
declare void @llvm.arm.neon.vst4.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32)
declare void @llvm.arm.neon.vst4.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32)
declare void @llvm.arm.neon.vst4.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32)
declare void @llvm.arm.neon.vst4.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32)
declare void @llvm.arm.neon.vst4.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32)
declare void @llvm.arm.neon.vst4.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, <1 x double>, i32)

define %struct.int8x16x2_t @test_vld1q_s8_x2(i8* %a)  {
; CHECK-LABEL: test_vld1q_s8_x2
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %1 = tail call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.vld1x2.v16i8(i8* %a, i32 1)
  %2 = extractvalue { <16 x i8>, <16 x i8> } %1, 0
  %3 = extractvalue { <16 x i8>, <16 x i8> } %1, 1
  %4 = insertvalue %struct.int8x16x2_t undef, <16 x i8> %2, 0, 0
  %5 = insertvalue %struct.int8x16x2_t %4, <16 x i8> %3, 0, 1
  ret %struct.int8x16x2_t %5
}

define %struct.int16x8x2_t @test_vld1q_s16_x2(i16* %a)  {
; CHECK-LABEL: test_vld1q_s16_x2
; CHECK: ld1 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x2.v8i16(i8* %1, i32 2)
  %3 = extractvalue { <8 x i16>, <8 x i16> } %2, 0
  %4 = extractvalue { <8 x i16>, <8 x i16> } %2, 1
  %5 = insertvalue %struct.int16x8x2_t undef, <8 x i16> %3, 0, 0
  %6 = insertvalue %struct.int16x8x2_t %5, <8 x i16> %4, 0, 1
  ret %struct.int16x8x2_t %6
}

define %struct.int32x4x2_t @test_vld1q_s32_x2(i32* %a)  {
; CHECK-LABEL: test_vld1q_s32_x2
; CHECK: ld1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %2 = tail call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.vld1x2.v4i32(i8* %1, i32 4)
  %3 = extractvalue { <4 x i32>, <4 x i32> } %2, 0
  %4 = extractvalue { <4 x i32>, <4 x i32> } %2, 1
  %5 = insertvalue %struct.int32x4x2_t undef, <4 x i32> %3, 0, 0
  %6 = insertvalue %struct.int32x4x2_t %5, <4 x i32> %4, 0, 1
  ret %struct.int32x4x2_t %6
}

define %struct.int64x2x2_t @test_vld1q_s64_x2(i64* %a)  {
; CHECK-LABEL: test_vld1q_s64_x2
; CHECK: ld1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %2 = tail call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.vld1x2.v2i64(i8* %1, i32 8)
  %3 = extractvalue { <2 x i64>, <2 x i64> } %2, 0
  %4 = extractvalue { <2 x i64>, <2 x i64> } %2, 1
  %5 = insertvalue %struct.int64x2x2_t undef, <2 x i64> %3, 0, 0
  %6 = insertvalue %struct.int64x2x2_t %5, <2 x i64> %4, 0, 1
  ret %struct.int64x2x2_t %6
}

define %struct.float32x4x2_t @test_vld1q_f32_x2(float* %a)  {
; CHECK-LABEL: test_vld1q_f32_x2
; CHECK: ld1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %2 = tail call { <4 x float>, <4 x float> } @llvm.aarch64.neon.vld1x2.v4f32(i8* %1, i32 4)
  %3 = extractvalue { <4 x float>, <4 x float> } %2, 0
  %4 = extractvalue { <4 x float>, <4 x float> } %2, 1
  %5 = insertvalue %struct.float32x4x2_t undef, <4 x float> %3, 0, 0
  %6 = insertvalue %struct.float32x4x2_t %5, <4 x float> %4, 0, 1
  ret %struct.float32x4x2_t %6
}


define %struct.float64x2x2_t @test_vld1q_f64_x2(double* %a)  {
; CHECK-LABEL: test_vld1q_f64_x2
; CHECK: ld1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %2 = tail call { <2 x double>, <2 x double> } @llvm.aarch64.neon.vld1x2.v2f64(i8* %1, i32 8)
  %3 = extractvalue { <2 x double>, <2 x double> } %2, 0
  %4 = extractvalue { <2 x double>, <2 x double> } %2, 1
  %5 = insertvalue %struct.float64x2x2_t undef, <2 x double> %3, 0, 0
  %6 = insertvalue %struct.float64x2x2_t %5, <2 x double> %4, 0, 1
  ret %struct.float64x2x2_t %6
}

define %struct.int8x8x2_t @test_vld1_s8_x2(i8* %a)  {
; CHECK-LABEL: test_vld1_s8_x2
; CHECK: ld1 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %1 = tail call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.vld1x2.v8i8(i8* %a, i32 1)
  %2 = extractvalue { <8 x i8>, <8 x i8> } %1, 0
  %3 = extractvalue { <8 x i8>, <8 x i8> } %1, 1
  %4 = insertvalue %struct.int8x8x2_t undef, <8 x i8> %2, 0, 0
  %5 = insertvalue %struct.int8x8x2_t %4, <8 x i8> %3, 0, 1
  ret %struct.int8x8x2_t %5
}

define %struct.int16x4x2_t @test_vld1_s16_x2(i16* %a)  {
; CHECK-LABEL: test_vld1_s16_x2
; CHECK: ld1 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.vld1x2.v4i16(i8* %1, i32 2)
  %3 = extractvalue { <4 x i16>, <4 x i16> } %2, 0
  %4 = extractvalue { <4 x i16>, <4 x i16> } %2, 1
  %5 = insertvalue %struct.int16x4x2_t undef, <4 x i16> %3, 0, 0
  %6 = insertvalue %struct.int16x4x2_t %5, <4 x i16> %4, 0, 1
  ret %struct.int16x4x2_t %6
}

define %struct.int32x2x2_t @test_vld1_s32_x2(i32* %a)  {
; CHECK-LABEL: test_vld1_s32_x2
; CHECK: ld1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %2 = tail call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.vld1x2.v2i32(i8* %1, i32 4)
  %3 = extractvalue { <2 x i32>, <2 x i32> } %2, 0
  %4 = extractvalue { <2 x i32>, <2 x i32> } %2, 1
  %5 = insertvalue %struct.int32x2x2_t undef, <2 x i32> %3, 0, 0
  %6 = insertvalue %struct.int32x2x2_t %5, <2 x i32> %4, 0, 1
  ret %struct.int32x2x2_t %6
}

define %struct.int64x1x2_t @test_vld1_s64_x2(i64* %a)  {
; CHECK-LABEL: test_vld1_s64_x2
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %2 = tail call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.vld1x2.v1i64(i8* %1, i32 8)
  %3 = extractvalue { <1 x i64>, <1 x i64> } %2, 0
  %4 = extractvalue { <1 x i64>, <1 x i64> } %2, 1
  %5 = insertvalue %struct.int64x1x2_t undef, <1 x i64> %3, 0, 0
  %6 = insertvalue %struct.int64x1x2_t %5, <1 x i64> %4, 0, 1
  ret %struct.int64x1x2_t %6
}

define %struct.float32x2x2_t @test_vld1_f32_x2(float* %a)  {
; CHECK-LABEL: test_vld1_f32_x2
; CHECK: ld1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %2 = tail call { <2 x float>, <2 x float> } @llvm.aarch64.neon.vld1x2.v2f32(i8* %1, i32 4)
  %3 = extractvalue { <2 x float>, <2 x float> } %2, 0
  %4 = extractvalue { <2 x float>, <2 x float> } %2, 1
  %5 = insertvalue %struct.float32x2x2_t undef, <2 x float> %3, 0, 0
  %6 = insertvalue %struct.float32x2x2_t %5, <2 x float> %4, 0, 1
  ret %struct.float32x2x2_t %6
}

define %struct.float64x1x2_t @test_vld1_f64_x2(double* %a)  {
; CHECK-LABEL: test_vld1_f64_x2
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %2 = tail call { <1 x double>, <1 x double> } @llvm.aarch64.neon.vld1x2.v1f64(i8* %1, i32 8)
  %3 = extractvalue { <1 x double>, <1 x double> } %2, 0
  %4 = extractvalue { <1 x double>, <1 x double> } %2, 1
  %5 = insertvalue %struct.float64x1x2_t undef, <1 x double> %3, 0, 0
  %6 = insertvalue %struct.float64x1x2_t %5, <1 x double> %4, 0, 1
  ret %struct.float64x1x2_t %6
}

define %struct.int8x16x3_t @test_vld1q_s8_x3(i8* %a)  {
; CHECK-LABEL: test_vld1q_s8_x3
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b},
; [{{x[0-9]+|sp}}]
  %1 = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.vld1x3.v16i8(i8* %a, i32 1)
  %2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %1, 0
  %3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %1, 1
  %4 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %1, 2
  %5 = insertvalue %struct.int8x16x3_t undef, <16 x i8> %2, 0, 0
  %6 = insertvalue %struct.int8x16x3_t %5, <16 x i8> %3, 0, 1
  %7 = insertvalue %struct.int8x16x3_t %6, <16 x i8> %4, 0, 2
  ret %struct.int8x16x3_t %7
}

define %struct.int16x8x3_t @test_vld1q_s16_x3(i16* %a)  {
; CHECK-LABEL: test_vld1q_s16_x3
; CHECK: ld1 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h},
; [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x3.v8i16(i8* %1, i32 2)
  %3 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %2, 0
  %4 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %2, 1
  %5 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %2, 2
  %6 = insertvalue %struct.int16x8x3_t undef, <8 x i16> %3, 0, 0
  %7 = insertvalue %struct.int16x8x3_t %6, <8 x i16> %4, 0, 1
  %8 = insertvalue %struct.int16x8x3_t %7, <8 x i16> %5, 0, 2
  ret %struct.int16x8x3_t %8
}

define %struct.int32x4x3_t @test_vld1q_s32_x3(i32* %a)  {
; CHECK-LABEL: test_vld1q_s32_x3
; CHECK: ld1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s},
; [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %2 = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.vld1x3.v4i32(i8* %1, i32 4)
  %3 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %2, 0
  %4 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %2, 1
  %5 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %2, 2
  %6 = insertvalue %struct.int32x4x3_t undef, <4 x i32> %3, 0, 0
  %7 = insertvalue %struct.int32x4x3_t %6, <4 x i32> %4, 0, 1
  %8 = insertvalue %struct.int32x4x3_t %7, <4 x i32> %5, 0, 2
  ret %struct.int32x4x3_t %8
}

define %struct.int64x2x3_t @test_vld1q_s64_x3(i64* %a)  {
; CHECK-LABEL: test_vld1q_s64_x3
; CHECK: ld1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d},
; [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %2 = tail call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.vld1x3.v2i64(i8* %1, i32 8)
  %3 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %2, 0
  %4 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %2, 1
  %5 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %2, 2
  %6 = insertvalue %struct.int64x2x3_t undef, <2 x i64> %3, 0, 0
  %7 = insertvalue %struct.int64x2x3_t %6, <2 x i64> %4, 0, 1
  %8 = insertvalue %struct.int64x2x3_t %7, <2 x i64> %5, 0, 2
  ret %struct.int64x2x3_t %8
}

define %struct.float32x4x3_t @test_vld1q_f32_x3(float* %a)  {
; CHECK-LABEL: test_vld1q_f32_x3
; CHECK: ld1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s},
; [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %2 = tail call { <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.vld1x3.v4f32(i8* %1, i32 4)
  %3 = extractvalue { <4 x float>, <4 x float>, <4 x float> } %2, 0
  %4 = extractvalue { <4 x float>, <4 x float>, <4 x float> } %2, 1
  %5 = extractvalue { <4 x float>, <4 x float>, <4 x float> } %2, 2
  %6 = insertvalue %struct.float32x4x3_t undef, <4 x float> %3, 0, 0
  %7 = insertvalue %struct.float32x4x3_t %6, <4 x float> %4, 0, 1
  %8 = insertvalue %struct.float32x4x3_t %7, <4 x float> %5, 0, 2
  ret %struct.float32x4x3_t %8
}


define %struct.float64x2x3_t @test_vld1q_f64_x3(double* %a)  {
; CHECK-LABEL: test_vld1q_f64_x3
; CHECK: ld1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d},
; [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %2 = tail call { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.vld1x3.v2f64(i8* %1, i32 8)
  %3 = extractvalue { <2 x double>, <2 x double>, <2 x double> } %2, 0
  %4 = extractvalue { <2 x double>, <2 x double>, <2 x double> } %2, 1
  %5 = extractvalue { <2 x double>, <2 x double>, <2 x double> } %2, 2
  %6 = insertvalue %struct.float64x2x3_t undef, <2 x double> %3, 0, 0
  %7 = insertvalue %struct.float64x2x3_t %6, <2 x double> %4, 0, 1
  %8 = insertvalue %struct.float64x2x3_t %7, <2 x double> %5, 0, 2
  ret %struct.float64x2x3_t %8
}

define %struct.int8x8x3_t @test_vld1_s8_x3(i8* %a)  {
; CHECK-LABEL: test_vld1_s8_x3
; CHECK: ld1 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b},
; [{{x[0-9]+|sp}}]
  %1 = tail call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.vld1x3.v8i8(i8* %a, i32 1)
  %2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %1, 0
  %3 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %1, 1
  %4 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %1, 2
  %5 = insertvalue %struct.int8x8x3_t undef, <8 x i8> %2, 0, 0
  %6 = insertvalue %struct.int8x8x3_t %5, <8 x i8> %3, 0, 1
  %7 = insertvalue %struct.int8x8x3_t %6, <8 x i8> %4, 0, 2
  ret %struct.int8x8x3_t %7
}

define %struct.int16x4x3_t @test_vld1_s16_x3(i16* %a)  {
; CHECK-LABEL: test_vld1_s16_x3
; CHECK: ld1 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h},
; [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.vld1x3.v4i16(i8* %1, i32 2)
  %3 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %2, 0
  %4 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %2, 1
  %5 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %2, 2
  %6 = insertvalue %struct.int16x4x3_t undef, <4 x i16> %3, 0, 0
  %7 = insertvalue %struct.int16x4x3_t %6, <4 x i16> %4, 0, 1
  %8 = insertvalue %struct.int16x4x3_t %7, <4 x i16> %5, 0, 2
  ret %struct.int16x4x3_t %8
}

define %struct.int32x2x3_t @test_vld1_s32_x3(i32* %a)  {
  %1 = bitcast i32* %a to i8*
; CHECK-LABEL: test_vld1_s32_x3
; CHECK: ld1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s},
; [{{x[0-9]+|sp}}]
  %2 = tail call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.vld1x3.v2i32(i8* %1, i32 4)
  %3 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %2, 0
  %4 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %2, 1
  %5 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %2, 2
  %6 = insertvalue %struct.int32x2x3_t undef, <2 x i32> %3, 0, 0
  %7 = insertvalue %struct.int32x2x3_t %6, <2 x i32> %4, 0, 1
  %8 = insertvalue %struct.int32x2x3_t %7, <2 x i32> %5, 0, 2
  ret %struct.int32x2x3_t %8
}

define %struct.int64x1x3_t @test_vld1_s64_x3(i64* %a)  {
; CHECK-LABEL: test_vld1_s64_x3
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d},
; [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %2 = tail call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.vld1x3.v1i64(i8* %1, i32 8)
  %3 = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %2, 0
  %4 = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %2, 1
  %5 = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %2, 2
  %6 = insertvalue %struct.int64x1x3_t undef, <1 x i64> %3, 0, 0
  %7 = insertvalue %struct.int64x1x3_t %6, <1 x i64> %4, 0, 1
  %8 = insertvalue %struct.int64x1x3_t %7, <1 x i64> %5, 0, 2
  ret %struct.int64x1x3_t %8
}

define %struct.float32x2x3_t @test_vld1_f32_x3(float* %a)  {
; CHECK-LABEL: test_vld1_f32_x3
; CHECK: ld1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s},
; [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %2 = tail call { <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.vld1x3.v2f32(i8* %1, i32 4)
  %3 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %2, 0
  %4 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %2, 1
  %5 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %2, 2
  %6 = insertvalue %struct.float32x2x3_t undef, <2 x float> %3, 0, 0
  %7 = insertvalue %struct.float32x2x3_t %6, <2 x float> %4, 0, 1
  %8 = insertvalue %struct.float32x2x3_t %7, <2 x float> %5, 0, 2
  ret %struct.float32x2x3_t %8
}


define %struct.float64x1x3_t @test_vld1_f64_x3(double* %a)  {
; CHECK-LABEL: test_vld1_f64_x3
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d},
; [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %2 = tail call { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.vld1x3.v1f64(i8* %1, i32 8)
  %3 = extractvalue { <1 x double>, <1 x double>, <1 x double> } %2, 0
  %4 = extractvalue { <1 x double>, <1 x double>, <1 x double> } %2, 1
  %5 = extractvalue { <1 x double>, <1 x double>, <1 x double> } %2, 2
  %6 = insertvalue %struct.float64x1x3_t undef, <1 x double> %3, 0, 0
  %7 = insertvalue %struct.float64x1x3_t %6, <1 x double> %4, 0, 1
  %8 = insertvalue %struct.float64x1x3_t %7, <1 x double> %5, 0, 2
  ret %struct.float64x1x3_t %8
}

define %struct.int8x16x4_t @test_vld1q_s8_x4(i8* %a)  {
; CHECK-LABEL: test_vld1q_s8_x4
; CHECK: ld1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b,
; v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %1 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.vld1x4.v16i8(i8* %a, i32 1)
  %2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %1, 0
  %3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %1, 1
  %4 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %1, 2
  %5 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %1, 3
  %6 = insertvalue %struct.int8x16x4_t undef, <16 x i8> %2, 0, 0
  %7 = insertvalue %struct.int8x16x4_t %6, <16 x i8> %3, 0, 1
  %8 = insertvalue %struct.int8x16x4_t %7, <16 x i8> %4, 0, 2
  %9 = insertvalue %struct.int8x16x4_t %8, <16 x i8> %5, 0, 3
  ret %struct.int8x16x4_t %9
}

define %struct.int16x8x4_t @test_vld1q_s16_x4(i16* %a)  {
; CHECK-LABEL: test_vld1q_s16_x4
; CHECK: ld1 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h,
; v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x4.v8i16(i8* %1, i32 2)
  %3 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %2, 0
  %4 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %2, 1
  %5 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %2, 2
  %6 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %2, 3
  %7 = insertvalue %struct.int16x8x4_t undef, <8 x i16> %3, 0, 0
  %8 = insertvalue %struct.int16x8x4_t %7, <8 x i16> %4, 0, 1
  %9 = insertvalue %struct.int16x8x4_t %8, <8 x i16> %5, 0, 2
  %10 = insertvalue %struct.int16x8x4_t %9, <8 x i16> %6, 0, 3
  ret %struct.int16x8x4_t %10
}

define %struct.int32x4x4_t @test_vld1q_s32_x4(i32* %a)  {
; CHECK-LABEL: test_vld1q_s32_x4
; CHECK: ld1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s,
; v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %2 = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.vld1x4.v4i32(i8* %1, i32 4)
  %3 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %2, 0
  %4 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %2, 1
  %5 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %2, 2
  %6 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %2, 3
  %7 = insertvalue %struct.int32x4x4_t undef, <4 x i32> %3, 0, 0
  %8 = insertvalue %struct.int32x4x4_t %7, <4 x i32> %4, 0, 1
  %9 = insertvalue %struct.int32x4x4_t %8, <4 x i32> %5, 0, 2
  %10 = insertvalue %struct.int32x4x4_t %9, <4 x i32> %6, 0, 3
  ret %struct.int32x4x4_t %10
}

define %struct.int64x2x4_t @test_vld1q_s64_x4(i64* %a)  {
; CHECK-LABEL: test_vld1q_s64_x4
; CHECK: ld1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d,
; v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %2 = tail call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.vld1x4.v2i64(i8* %1, i32 8)
  %3 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %2, 0
  %4 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %2, 1
  %5 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %2, 2
  %6 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %2, 3
  %7 = insertvalue %struct.int64x2x4_t undef, <2 x i64> %3, 0, 0
  %8 = insertvalue %struct.int64x2x4_t %7, <2 x i64> %4, 0, 1
  %9 = insertvalue %struct.int64x2x4_t %8, <2 x i64> %5, 0, 2
  %10 = insertvalue %struct.int64x2x4_t %9, <2 x i64> %6, 0, 3
  ret %struct.int64x2x4_t %10
}

define %struct.float32x4x4_t @test_vld1q_f32_x4(float* %a)  {
; CHECK-LABEL: test_vld1q_f32_x4
; CHECK: ld1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s,
; v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %2 = tail call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.vld1x4.v4f32(i8* %1, i32 4)
  %3 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 0
  %4 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 1
  %5 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 2
  %6 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 3
  %7 = insertvalue %struct.float32x4x4_t undef, <4 x float> %3, 0, 0
  %8 = insertvalue %struct.float32x4x4_t %7, <4 x float> %4, 0, 1
  %9 = insertvalue %struct.float32x4x4_t %8, <4 x float> %5, 0, 2
  %10 = insertvalue %struct.float32x4x4_t %9, <4 x float> %6, 0, 3
  ret %struct.float32x4x4_t %10
}

define %struct.float64x2x4_t @test_vld1q_f64_x4(double* %a)  {
; CHECK-LABEL: test_vld1q_f64_x4
; CHECK: ld1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d,
; v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %2 = tail call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.vld1x4.v2f64(i8* %1, i32 8)
  %3 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %2, 0
  %4 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %2, 1
  %5 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %2, 2
  %6 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %2, 3
  %7 = insertvalue %struct.float64x2x4_t undef, <2 x double> %3, 0, 0
  %8 = insertvalue %struct.float64x2x4_t %7, <2 x double> %4, 0, 1
  %9 = insertvalue %struct.float64x2x4_t %8, <2 x double> %5, 0, 2
  %10 = insertvalue %struct.float64x2x4_t %9, <2 x double> %6, 0, 3
  ret %struct.float64x2x4_t %10
}

define %struct.int8x8x4_t @test_vld1_s8_x4(i8* %a)  {
; CHECK-LABEL: test_vld1_s8_x4
; CHECK: ld1 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b,
; v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %1 = tail call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.vld1x4.v8i8(i8* %a, i32 1)
  %2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %1, 0
  %3 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %1, 1
  %4 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %1, 2
  %5 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %1, 3
  %6 = insertvalue %struct.int8x8x4_t undef, <8 x i8> %2, 0, 0
  %7 = insertvalue %struct.int8x8x4_t %6, <8 x i8> %3, 0, 1
  %8 = insertvalue %struct.int8x8x4_t %7, <8 x i8> %4, 0, 2
  %9 = insertvalue %struct.int8x8x4_t %8, <8 x i8> %5, 0, 3
  ret %struct.int8x8x4_t %9
}

define %struct.int16x4x4_t @test_vld1_s16_x4(i16* %a)  {
; CHECK-LABEL: test_vld1_s16_x4
; CHECK: ld1 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h,
; v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.vld1x4.v4i16(i8* %1, i32 2)
  %3 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %2, 0
  %4 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %2, 1
  %5 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %2, 2
  %6 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %2, 3
  %7 = insertvalue %struct.int16x4x4_t undef, <4 x i16> %3, 0, 0
  %8 = insertvalue %struct.int16x4x4_t %7, <4 x i16> %4, 0, 1
  %9 = insertvalue %struct.int16x4x4_t %8, <4 x i16> %5, 0, 2
  %10 = insertvalue %struct.int16x4x4_t %9, <4 x i16> %6, 0, 3
  ret %struct.int16x4x4_t %10
}

define %struct.int32x2x4_t @test_vld1_s32_x4(i32* %a)  {
; CHECK-LABEL: test_vld1_s32_x4
; CHECK: ld1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s,
; v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %2 = tail call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.vld1x4.v2i32(i8* %1, i32 4)
  %3 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %2, 0
  %4 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %2, 1
  %5 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %2, 2
  %6 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %2, 3
  %7 = insertvalue %struct.int32x2x4_t undef, <2 x i32> %3, 0, 0
  %8 = insertvalue %struct.int32x2x4_t %7, <2 x i32> %4, 0, 1
  %9 = insertvalue %struct.int32x2x4_t %8, <2 x i32> %5, 0, 2
  %10 = insertvalue %struct.int32x2x4_t %9, <2 x i32> %6, 0, 3
  ret %struct.int32x2x4_t %10
}

define %struct.int64x1x4_t @test_vld1_s64_x4(i64* %a)  {
; CHECK-LABEL: test_vld1_s64_x4
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d,
; v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %2 = tail call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.vld1x4.v1i64(i8* %1, i32 8)
  %3 = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %2, 0
  %4 = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %2, 1
  %5 = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %2, 2
  %6 = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %2, 3
  %7 = insertvalue %struct.int64x1x4_t undef, <1 x i64> %3, 0, 0
  %8 = insertvalue %struct.int64x1x4_t %7, <1 x i64> %4, 0, 1
  %9 = insertvalue %struct.int64x1x4_t %8, <1 x i64> %5, 0, 2
  %10 = insertvalue %struct.int64x1x4_t %9, <1 x i64> %6, 0, 3
  ret %struct.int64x1x4_t %10
}

define %struct.float32x2x4_t @test_vld1_f32_x4(float* %a)  {
; CHECK-LABEL: test_vld1_f32_x4
; CHECK: ld1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s,
; v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %2 = tail call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.vld1x4.v2f32(i8* %1, i32 4)
  %3 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %2, 0
  %4 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %2, 1
  %5 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %2, 2
  %6 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %2, 3
  %7 = insertvalue %struct.float32x2x4_t undef, <2 x float> %3, 0, 0
  %8 = insertvalue %struct.float32x2x4_t %7, <2 x float> %4, 0, 1
  %9 = insertvalue %struct.float32x2x4_t %8, <2 x float> %5, 0, 2
  %10 = insertvalue %struct.float32x2x4_t %9, <2 x float> %6, 0, 3
  ret %struct.float32x2x4_t %10
}


define %struct.float64x1x4_t @test_vld1_f64_x4(double* %a)  {
; CHECK-LABEL: test_vld1_f64_x4
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d,
; v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %2 = tail call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.vld1x4.v1f64(i8* %1, i32 8)
  %3 = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %2, 0
  %4 = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %2, 1
  %5 = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %2, 2
  %6 = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %2, 3
  %7 = insertvalue %struct.float64x1x4_t undef, <1 x double> %3, 0, 0
  %8 = insertvalue %struct.float64x1x4_t %7, <1 x double> %4, 0, 1
  %9 = insertvalue %struct.float64x1x4_t %8, <1 x double> %5, 0, 2
  %10 = insertvalue %struct.float64x1x4_t %9, <1 x double> %6, 0, 3
  ret %struct.float64x1x4_t %10
}

define void @test_vst1q_s8_x2(i8* %a, [2 x <16 x i8>] %b)  {
; CHECK-LABEL: test_vst1q_s8_x2
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <16 x i8>] %b, 0
  %2 = extractvalue [2 x <16 x i8>] %b, 1
  tail call void @llvm.aarch64.neon.vst1x2.v16i8(i8* %a, <16 x i8> %1, <16 x i8> %2, i32 1)
  ret void
}

define void @test_vst1q_s16_x2(i16* %a, [2 x <8 x i16>] %b)  {
; CHECK-LABEL: test_vst1q_s16_x2
; CHECK: st1 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <8 x i16>] %b, 0
  %2 = extractvalue [2 x <8 x i16>] %b, 1
  %3 = bitcast i16* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v8i16(i8* %3, <8 x i16> %1, <8 x i16> %2, i32 2)
  ret void
}

define void @test_vst1q_s32_x2(i32* %a, [2 x <4 x i32>] %b)  {
; CHECK-LABEL: test_vst1q_s32_x2
; CHECK: st1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <4 x i32>] %b, 0
  %2 = extractvalue [2 x <4 x i32>] %b, 1
  %3 = bitcast i32* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v4i32(i8* %3, <4 x i32> %1, <4 x i32> %2, i32 4)
  ret void
}

define void @test_vst1q_s64_x2(i64* %a, [2 x <2 x i64>] %b)  {
; CHECK-LABEL: test_vst1q_s64_x2
; CHECK: st1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <2 x i64>] %b, 0
  %2 = extractvalue [2 x <2 x i64>] %b, 1
  %3 = bitcast i64* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v2i64(i8* %3, <2 x i64> %1, <2 x i64> %2, i32 8)
  ret void
}

define void @test_vst1q_f32_x2(float* %a, [2 x <4 x float>] %b)  {
; CHECK-LABEL: test_vst1q_f32_x2
; CHECK: st1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <4 x float>] %b, 0
  %2 = extractvalue [2 x <4 x float>] %b, 1
  %3 = bitcast float* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v4f32(i8* %3, <4 x float> %1, <4 x float> %2, i32 4)
  ret void
}


define void @test_vst1q_f64_x2(double* %a, [2 x <2 x double>] %b)  {
; CHECK-LABEL: test_vst1q_f64_x2
; CHECK: st1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <2 x double>] %b, 0
  %2 = extractvalue [2 x <2 x double>] %b, 1
  %3 = bitcast double* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v2f64(i8* %3, <2 x double> %1, <2 x double> %2, i32 8)
  ret void
}

define void @test_vst1_s8_x2(i8* %a, [2 x <8 x i8>] %b)  {
; CHECK-LABEL: test_vst1_s8_x2
; CHECK: st1 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <8 x i8>] %b, 0
  %2 = extractvalue [2 x <8 x i8>] %b, 1
  tail call void @llvm.aarch64.neon.vst1x2.v8i8(i8* %a, <8 x i8> %1, <8 x i8> %2, i32 1)
  ret void
}

define void @test_vst1_s16_x2(i16* %a, [2 x <4 x i16>] %b)  {
; CHECK-LABEL: test_vst1_s16_x2
; CHECK: st1 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <4 x i16>] %b, 0
  %2 = extractvalue [2 x <4 x i16>] %b, 1
  %3 = bitcast i16* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v4i16(i8* %3, <4 x i16> %1, <4 x i16> %2, i32 2)
  ret void
}

define void @test_vst1_s32_x2(i32* %a, [2 x <2 x i32>] %b)  {
; CHECK-LABEL: test_vst1_s32_x2
; CHECK: st1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <2 x i32>] %b, 0
  %2 = extractvalue [2 x <2 x i32>] %b, 1
  %3 = bitcast i32* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v2i32(i8* %3, <2 x i32> %1, <2 x i32> %2, i32 4)
  ret void
}

define void @test_vst1_s64_x2(i64* %a, [2 x <1 x i64>] %b)  {
; CHECK-LABEL: test_vst1_s64_x2
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <1 x i64>] %b, 0
  %2 = extractvalue [2 x <1 x i64>] %b, 1
  %3 = bitcast i64* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v1i64(i8* %3, <1 x i64> %1, <1 x i64> %2, i32 8)
  ret void
}

define void @test_vst1_f32_x2(float* %a, [2 x <2 x float>] %b)  {
; CHECK-LABEL: test_vst1_f32_x2
; CHECK: st1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <2 x float>] %b, 0
  %2 = extractvalue [2 x <2 x float>] %b, 1
  %3 = bitcast float* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v2f32(i8* %3, <2 x float> %1, <2 x float> %2, i32 4)
  ret void
}

define void @test_vst1_f64_x2(double* %a, [2 x <1 x double>] %b)  {
; CHECK-LABEL: test_vst1_f64_x2
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [2 x <1 x double>] %b, 0
  %2 = extractvalue [2 x <1 x double>] %b, 1
  %3 = bitcast double* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v1f64(i8* %3, <1 x double> %1, <1 x double> %2, i32 8)
  ret void
}

define void @test_vst1q_s8_x3(i8* %a, [3 x <16 x i8>] %b)  {
; CHECK-LABEL: test_vst1q_s8_x3
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <16 x i8>] %b, 0
  %2 = extractvalue [3 x <16 x i8>] %b, 1
  %3 = extractvalue [3 x <16 x i8>] %b, 2
  tail call void @llvm.aarch64.neon.vst1x3.v16i8(i8* %a, <16 x i8> %1, <16 x i8> %2, <16 x i8> %3, i32 1)
  ret void
}

define void @test_vst1q_s16_x3(i16* %a, [3 x <8 x i16>] %b)  {
; CHECK-LABEL: test_vst1q_s16_x3
; CHECK: st1 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <8 x i16>] %b, 0
  %2 = extractvalue [3 x <8 x i16>] %b, 1
  %3 = extractvalue [3 x <8 x i16>] %b, 2
  %4 = bitcast i16* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v8i16(i8* %4, <8 x i16> %1, <8 x i16> %2, <8 x i16> %3, i32 2)
  ret void
}

define void @test_vst1q_s32_x3(i32* %a, [3 x <4 x i32>] %b)  {
; CHECK-LABEL: test_vst1q_s32_x3
; CHECK: st1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <4 x i32>] %b, 0
  %2 = extractvalue [3 x <4 x i32>] %b, 1
  %3 = extractvalue [3 x <4 x i32>] %b, 2
  %4 = bitcast i32* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v4i32(i8* %4, <4 x i32> %1, <4 x i32> %2, <4 x i32> %3, i32 4)
  ret void
}

define void @test_vst1q_s64_x3(i64* %a, [3 x <2 x i64>] %b)  {
; CHECK-LABEL: test_vst1q_s64_x3
; CHECK: st1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <2 x i64>] %b, 0
  %2 = extractvalue [3 x <2 x i64>] %b, 1
  %3 = extractvalue [3 x <2 x i64>] %b, 2
  %4 = bitcast i64* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v2i64(i8* %4, <2 x i64> %1, <2 x i64> %2, <2 x i64> %3, i32 8)
  ret void
}

define void @test_vst1q_f32_x3(float* %a, [3 x <4 x float>] %b)  {
; CHECK-LABEL: test_vst1q_f32_x3
; CHECK: st1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <4 x float>] %b, 0
  %2 = extractvalue [3 x <4 x float>] %b, 1
  %3 = extractvalue [3 x <4 x float>] %b, 2
  %4 = bitcast float* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v4f32(i8* %4, <4 x float> %1, <4 x float> %2, <4 x float> %3, i32 4)
  ret void
}

define void @test_vst1q_f64_x3(double* %a, [3 x <2 x double>] %b)  {
; CHECK-LABEL: test_vst1q_f64_x3
; CHECK: st1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <2 x double>] %b, 0
  %2 = extractvalue [3 x <2 x double>] %b, 1
  %3 = extractvalue [3 x <2 x double>] %b, 2
  %4 = bitcast double* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v2f64(i8* %4, <2 x double> %1, <2 x double> %2, <2 x double> %3, i32 8)
  ret void
}

define void @test_vst1_s8_x3(i8* %a, [3 x <8 x i8>] %b)  {
; CHECK-LABEL: test_vst1_s8_x3
; CHECK: st1 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <8 x i8>] %b, 0
  %2 = extractvalue [3 x <8 x i8>] %b, 1
  %3 = extractvalue [3 x <8 x i8>] %b, 2
  tail call void @llvm.aarch64.neon.vst1x3.v8i8(i8* %a, <8 x i8> %1, <8 x i8> %2, <8 x i8> %3, i32 1)
  ret void
}

define void @test_vst1_s16_x3(i16* %a, [3 x <4 x i16>] %b)  {
; CHECK-LABEL: test_vst1_s16_x3
; CHECK: st1 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <4 x i16>] %b, 0
  %2 = extractvalue [3 x <4 x i16>] %b, 1
  %3 = extractvalue [3 x <4 x i16>] %b, 2
  %4 = bitcast i16* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v4i16(i8* %4, <4 x i16> %1, <4 x i16> %2, <4 x i16> %3, i32 2)
  ret void
}

define void @test_vst1_s32_x3(i32* %a, [3 x <2 x i32>] %b)  {
; CHECK-LABEL: test_vst1_s32_x3
; CHECK: st1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <2 x i32>] %b, 0
  %2 = extractvalue [3 x <2 x i32>] %b, 1
  %3 = extractvalue [3 x <2 x i32>] %b, 2
  %4 = bitcast i32* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v2i32(i8* %4, <2 x i32> %1, <2 x i32> %2, <2 x i32> %3, i32 4)
  ret void
}

define void @test_vst1_s64_x3(i64* %a, [3 x <1 x i64>] %b)  {
; CHECK-LABEL: test_vst1_s64_x3
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <1 x i64>] %b, 0
  %2 = extractvalue [3 x <1 x i64>] %b, 1
  %3 = extractvalue [3 x <1 x i64>] %b, 2
  %4 = bitcast i64* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v1i64(i8* %4, <1 x i64> %1, <1 x i64> %2, <1 x i64> %3, i32 8)
  ret void
}

define void @test_vst1_f32_x3(float* %a, [3 x <2 x float>] %b)  {
; CHECK-LABEL: test_vst1_f32_x3
; CHECK: st1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <2 x float>] %b, 0
  %2 = extractvalue [3 x <2 x float>] %b, 1
  %3 = extractvalue [3 x <2 x float>] %b, 2
  %4 = bitcast float* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v2f32(i8* %4, <2 x float> %1, <2 x float> %2, <2 x float> %3, i32 4)
  ret void
}

define void @test_vst1_f64_x3(double* %a, [3 x <1 x double>] %b)  {
; CHECK-LABEL: test_vst1_f64_x3
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d},
; [{{x[0-9]+|sp}}]
  %1 = extractvalue [3 x <1 x double>] %b, 0
  %2 = extractvalue [3 x <1 x double>] %b, 1
  %3 = extractvalue [3 x <1 x double>] %b, 2
  %4 = bitcast double* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v1f64(i8* %4, <1 x double> %1, <1 x double> %2, <1 x double> %3, i32 8)
  ret void
}

define void @test_vst1q_s8_x4(i8* %a, [4 x <16 x i8>] %b)  {
; CHECK-LABEL: test_vst1q_s8_x4
; CHECK: st1 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b,
; v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <16 x i8>] %b, 0
  %2 = extractvalue [4 x <16 x i8>] %b, 1
  %3 = extractvalue [4 x <16 x i8>] %b, 2
  %4 = extractvalue [4 x <16 x i8>] %b, 3
  tail call void @llvm.aarch64.neon.vst1x4.v16i8(i8* %a, <16 x i8> %1, <16 x i8> %2, <16 x i8> %3, <16 x i8> %4, i32 1)
  ret void
}

define void @test_vst1q_s16_x4(i16* %a, [4 x <8 x i16>] %b)  {
; CHECK-LABEL: test_vst1q_s16_x4
; CHECK: st1 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h,
; v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <8 x i16>] %b, 0
  %2 = extractvalue [4 x <8 x i16>] %b, 1
  %3 = extractvalue [4 x <8 x i16>] %b, 2
  %4 = extractvalue [4 x <8 x i16>] %b, 3
  %5 = bitcast i16* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v8i16(i8* %5, <8 x i16> %1, <8 x i16> %2, <8 x i16> %3, <8 x i16> %4, i32 2)
  ret void
}

define void @test_vst1q_s32_x4(i32* %a, [4 x <4 x i32>] %b)  {
; CHECK-LABEL: test_vst1q_s32_x4
; CHECK: st1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s,
; v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <4 x i32>] %b, 0
  %2 = extractvalue [4 x <4 x i32>] %b, 1
  %3 = extractvalue [4 x <4 x i32>] %b, 2
  %4 = extractvalue [4 x <4 x i32>] %b, 3
  %5 = bitcast i32* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v4i32(i8* %5, <4 x i32> %1, <4 x i32> %2, <4 x i32> %3, <4 x i32> %4, i32 4)
  ret void
}

define void @test_vst1q_s64_x4(i64* %a, [4 x <2 x i64>] %b)  {
; CHECK-LABEL: test_vst1q_s64_x4
; CHECK: st1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d,
; v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <2 x i64>] %b, 0
  %2 = extractvalue [4 x <2 x i64>] %b, 1
  %3 = extractvalue [4 x <2 x i64>] %b, 2
  %4 = extractvalue [4 x <2 x i64>] %b, 3
  %5 = bitcast i64* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v2i64(i8* %5, <2 x i64> %1, <2 x i64> %2, <2 x i64> %3, <2 x i64> %4, i32 8)
  ret void
}

define void @test_vst1q_f32_x4(float* %a, [4 x <4 x float>] %b)  {
; CHECK-LABEL: test_vst1q_f32_x4
; CHECK: st1 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s,
; v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <4 x float>] %b, 0
  %2 = extractvalue [4 x <4 x float>] %b, 1
  %3 = extractvalue [4 x <4 x float>] %b, 2
  %4 = extractvalue [4 x <4 x float>] %b, 3
  %5 = bitcast float* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v4f32(i8* %5, <4 x float> %1, <4 x float> %2, <4 x float> %3, <4 x float> %4, i32 4)
  ret void
}

define void @test_vst1q_f64_x4(double* %a, [4 x <2 x double>] %b)  {
; CHECK-LABEL: test_vst1q_f64_x4
; CHECK: st1 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d,
; v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <2 x double>] %b, 0
  %2 = extractvalue [4 x <2 x double>] %b, 1
  %3 = extractvalue [4 x <2 x double>] %b, 2
  %4 = extractvalue [4 x <2 x double>] %b, 3
  %5 = bitcast double* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v2f64(i8* %5, <2 x double> %1, <2 x double> %2, <2 x double> %3, <2 x double> %4, i32 8)
  ret void
}

define void @test_vst1_s8_x4(i8* %a, [4 x <8 x i8>] %b)  {
; CHECK-LABEL: test_vst1_s8_x4
; CHECK: st1 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b,
; v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <8 x i8>] %b, 0
  %2 = extractvalue [4 x <8 x i8>] %b, 1
  %3 = extractvalue [4 x <8 x i8>] %b, 2
  %4 = extractvalue [4 x <8 x i8>] %b, 3
  tail call void @llvm.aarch64.neon.vst1x4.v8i8(i8* %a, <8 x i8> %1, <8 x i8> %2, <8 x i8> %3, <8 x i8> %4, i32 1)
  ret void
}

define void @test_vst1_s16_x4(i16* %a, [4 x <4 x i16>] %b)  {
; CHECK-LABEL: test_vst1_s16_x4
; CHECK: st1 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h,
; v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <4 x i16>] %b, 0
  %2 = extractvalue [4 x <4 x i16>] %b, 1
  %3 = extractvalue [4 x <4 x i16>] %b, 2
  %4 = extractvalue [4 x <4 x i16>] %b, 3
  %5 = bitcast i16* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v4i16(i8* %5, <4 x i16> %1, <4 x i16> %2, <4 x i16> %3, <4 x i16> %4, i32 2)
  ret void
}

define void @test_vst1_s32_x4(i32* %a, [4 x <2 x i32>] %b)  {
; CHECK-LABEL: test_vst1_s32_x4
; CHECK: st1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s,
; v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <2 x i32>] %b, 0
  %2 = extractvalue [4 x <2 x i32>] %b, 1
  %3 = extractvalue [4 x <2 x i32>] %b, 2
  %4 = extractvalue [4 x <2 x i32>] %b, 3
  %5 = bitcast i32* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v2i32(i8* %5, <2 x i32> %1, <2 x i32> %2, <2 x i32> %3, <2 x i32> %4, i32 4)
  ret void
}

define void @test_vst1_s64_x4(i64* %a, [4 x <1 x i64>] %b)  {
; CHECK-LABEL: test_vst1_s64_x4
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d,
; v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <1 x i64>] %b, 0
  %2 = extractvalue [4 x <1 x i64>] %b, 1
  %3 = extractvalue [4 x <1 x i64>] %b, 2
  %4 = extractvalue [4 x <1 x i64>] %b, 3
  %5 = bitcast i64* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v1i64(i8* %5, <1 x i64> %1, <1 x i64> %2, <1 x i64> %3, <1 x i64> %4, i32 8)
  ret void
}

define void @test_vst1_f32_x4(float* %a, [4 x <2 x float>] %b)  {
; CHECK-LABEL: test_vst1_f32_x4
; CHECK: st1 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s,
; v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <2 x float>] %b, 0
  %2 = extractvalue [4 x <2 x float>] %b, 1
  %3 = extractvalue [4 x <2 x float>] %b, 2
  %4 = extractvalue [4 x <2 x float>] %b, 3
  %5 = bitcast float* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v2f32(i8* %5, <2 x float> %1, <2 x float> %2, <2 x float> %3, <2 x float> %4, i32 4)
  ret void
}

define void @test_vst1_f64_x4(double* %a, [4 x <1 x double>] %b)  {
; CHECK-LABEL: test_vst1_f64_x4
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d,
; v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = extractvalue [4 x <1 x double>] %b, 0
  %2 = extractvalue [4 x <1 x double>] %b, 1
  %3 = extractvalue [4 x <1 x double>] %b, 2
  %4 = extractvalue [4 x <1 x double>] %b, 3
  %5 = bitcast double* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v1f64(i8* %5, <1 x double> %1, <1 x double> %2, <1 x double> %3, <1 x double> %4, i32 8)
  ret void
}

declare { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.vld1x2.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x2.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.vld1x2.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.vld1x2.v2i64(i8*, i32)
declare { <4 x float>, <4 x float> } @llvm.aarch64.neon.vld1x2.v4f32(i8*, i32)
declare { <2 x double>, <2 x double> } @llvm.aarch64.neon.vld1x2.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.vld1x2.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.vld1x2.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.vld1x2.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.vld1x2.v1i64(i8*, i32)
declare { <2 x float>, <2 x float> } @llvm.aarch64.neon.vld1x2.v2f32(i8*, i32)
declare { <1 x double>, <1 x double> } @llvm.aarch64.neon.vld1x2.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.vld1x3.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x3.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.vld1x3.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.vld1x3.v2i64(i8*, i32)
declare { <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.vld1x3.v4f32(i8*, i32)
declare { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.vld1x3.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.vld1x3.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.vld1x3.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.vld1x3.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.vld1x3.v1i64(i8*, i32)
declare { <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.vld1x3.v2f32(i8*, i32)
declare { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.vld1x3.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.vld1x4.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x4.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.vld1x4.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.vld1x4.v2i64(i8*, i32)
declare { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.vld1x4.v4f32(i8*, i32)
declare { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.vld1x4.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.vld1x4.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.vld1x4.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.vld1x4.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.vld1x4.v1i64(i8*, i32)
declare { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.vld1x4.v2f32(i8*, i32)
declare { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.vld1x4.v1f64(i8*, i32)
declare void @llvm.aarch64.neon.vst1x2.v16i8(i8*, <16 x i8>, <16 x i8>, i32)
declare void @llvm.aarch64.neon.vst1x2.v8i16(i8*, <8 x i16>, <8 x i16>, i32)
declare void @llvm.aarch64.neon.vst1x2.v4i32(i8*, <4 x i32>, <4 x i32>, i32)
declare void @llvm.aarch64.neon.vst1x2.v2i64(i8*, <2 x i64>, <2 x i64>, i32)
declare void @llvm.aarch64.neon.vst1x2.v4f32(i8*, <4 x float>, <4 x float>, i32)
declare void @llvm.aarch64.neon.vst1x2.v2f64(i8*, <2 x double>, <2 x double>, i32)
declare void @llvm.aarch64.neon.vst1x2.v8i8(i8*, <8 x i8>, <8 x i8>, i32)
declare void @llvm.aarch64.neon.vst1x2.v4i16(i8*, <4 x i16>, <4 x i16>, i32)
declare void @llvm.aarch64.neon.vst1x2.v2i32(i8*, <2 x i32>, <2 x i32>, i32)
declare void @llvm.aarch64.neon.vst1x2.v1i64(i8*, <1 x i64>, <1 x i64>, i32)
declare void @llvm.aarch64.neon.vst1x2.v2f32(i8*, <2 x float>, <2 x float>, i32)
declare void @llvm.aarch64.neon.vst1x2.v1f64(i8*, <1 x double>, <1 x double>, i32)
declare void @llvm.aarch64.neon.vst1x3.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, i32)
declare void @llvm.aarch64.neon.vst1x3.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32)
declare void @llvm.aarch64.neon.vst1x3.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32)
declare void @llvm.aarch64.neon.vst1x3.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, i32)
declare void @llvm.aarch64.neon.vst1x3.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, i32)
declare void @llvm.aarch64.neon.vst1x3.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, i32)
declare void @llvm.aarch64.neon.vst1x3.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32)
declare void @llvm.aarch64.neon.vst1x3.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32)
declare void @llvm.aarch64.neon.vst1x3.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32)
declare void @llvm.aarch64.neon.vst1x3.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, i32)
declare void @llvm.aarch64.neon.vst1x3.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32)
declare void @llvm.aarch64.neon.vst1x3.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, i32)
declare void @llvm.aarch64.neon.vst1x4.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32)
declare void @llvm.aarch64.neon.vst1x4.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32)
declare void @llvm.aarch64.neon.vst1x4.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32)
declare void @llvm.aarch64.neon.vst1x4.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, i32)
declare void @llvm.aarch64.neon.vst1x4.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32)
declare void @llvm.aarch64.neon.vst1x4.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, <2 x double>, i32)
declare void @llvm.aarch64.neon.vst1x4.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32)
declare void @llvm.aarch64.neon.vst1x4.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32)
declare void @llvm.aarch64.neon.vst1x4.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32)
declare void @llvm.aarch64.neon.vst1x4.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32)
declare void @llvm.aarch64.neon.vst1x4.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32)
declare void @llvm.aarch64.neon.vst1x4.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, <1 x double>, i32)
