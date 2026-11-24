; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -force-streaming-compatible < %s | FileCheck %s --check-prefix=SVE

define void @interleave_intrinsic_64_i8(<8 x i8> %v0, <8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3, <8 x i8> %v4, <8 x i8> %v5, <8 x i8> %v6, <8 x i8> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_i8:
  %wide = call <64 x i8> @llvm.vector.interleave8.v64i8(<8 x i8> %v0, <8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3, <8 x i8> %v4, <8 x i8> %v5, <8 x i8> %v6, <8 x i8> %v7)
  store <64 x i8> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_i8(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_i8:
  %wide = load <64 x i8>, ptr %src, align 8
  %parts = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.vector.deinterleave8.v64i8(<64 x i8> %wide)
  %v0 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 0
  store <8 x i8> %v0, ptr %dst, align 8
  %v1 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <8 x i8> %v1, ptr %d1, align 8
  %v2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <8 x i8> %v2, ptr %d2, align 8
  %v3 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <8 x i8> %v3, ptr %d3, align 8
  %v4 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <8 x i8> %v4, ptr %d4, align 8
  %v5 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <8 x i8> %v5, ptr %d5, align 8
  %v6 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 48
  store <8 x i8> %v6, ptr %d6, align 8
  %v7 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 56
  store <8 x i8> %v7, ptr %d7, align 8
  ret void
}

define void @interleave_intrinsic_64_i16(<4 x i16> %v0, <4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3, <4 x i16> %v4, <4 x i16> %v5, <4 x i16> %v6, <4 x i16> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_i16:
  %wide = call <32 x i16> @llvm.vector.interleave8.v32i16(<4 x i16> %v0, <4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3, <4 x i16> %v4, <4 x i16> %v5, <4 x i16> %v6, <4 x i16> %v7)
  store <32 x i16> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_i16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_i16:
  %wide = load <32 x i16>, ptr %src, align 8
  %parts = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.vector.deinterleave8.v32i16(<32 x i16> %wide)
  %v0 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 0
  store <4 x i16> %v0, ptr %dst, align 8
  %v1 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <4 x i16> %v1, ptr %d1, align 8
  %v2 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <4 x i16> %v2, ptr %d2, align 8
  %v3 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <4 x i16> %v3, ptr %d3, align 8
  %v4 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <4 x i16> %v4, ptr %d4, align 8
  %v5 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <4 x i16> %v5, ptr %d5, align 8
  %v6 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 48
  store <4 x i16> %v6, ptr %d6, align 8
  %v7 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 56
  store <4 x i16> %v7, ptr %d7, align 8
  ret void
}

define void @interleave_intrinsic_64_i32(<2 x i32> %v0, <2 x i32> %v1, <2 x i32> %v2, <2 x i32> %v3, <2 x i32> %v4, <2 x i32> %v5, <2 x i32> %v6, <2 x i32> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_i32:
  %wide = call <16 x i32> @llvm.vector.interleave8.v16i32(<2 x i32> %v0, <2 x i32> %v1, <2 x i32> %v2, <2 x i32> %v3, <2 x i32> %v4, <2 x i32> %v5, <2 x i32> %v6, <2 x i32> %v7)
  store <16 x i32> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_i32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_i32:
  %wide = load <16 x i32>, ptr %src, align 8
  %parts = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.vector.deinterleave8.v16i32(<16 x i32> %wide)
  %v0 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 0
  store <2 x i32> %v0, ptr %dst, align 8
  %v1 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <2 x i32> %v1, ptr %d1, align 8
  %v2 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <2 x i32> %v2, ptr %d2, align 8
  %v3 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <2 x i32> %v3, ptr %d3, align 8
  %v4 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <2 x i32> %v4, ptr %d4, align 8
  %v5 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <2 x i32> %v5, ptr %d5, align 8
  %v6 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 48
  store <2 x i32> %v6, ptr %d6, align 8
  %v7 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 56
  store <2 x i32> %v7, ptr %d7, align 8
  ret void
}

define void @interleave_intrinsic_64_f16(<4 x half> %v0, <4 x half> %v1, <4 x half> %v2, <4 x half> %v3, <4 x half> %v4, <4 x half> %v5, <4 x half> %v6, <4 x half> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_f16:
  %wide = call <32 x half> @llvm.vector.interleave8.v32f16(<4 x half> %v0, <4 x half> %v1, <4 x half> %v2, <4 x half> %v3, <4 x half> %v4, <4 x half> %v5, <4 x half> %v6, <4 x half> %v7)
  store <32 x half> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_f16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_f16:
  %wide = load <32 x half>, ptr %src, align 8
  %parts = call { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.vector.deinterleave8.v32f16(<32 x half> %wide)
  %v0 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 0
  store <4 x half> %v0, ptr %dst, align 8
  %v1 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <4 x half> %v1, ptr %d1, align 8
  %v2 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <4 x half> %v2, ptr %d2, align 8
  %v3 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <4 x half> %v3, ptr %d3, align 8
  %v4 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <4 x half> %v4, ptr %d4, align 8
  %v5 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <4 x half> %v5, ptr %d5, align 8
  %v6 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 48
  store <4 x half> %v6, ptr %d6, align 8
  %v7 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 56
  store <4 x half> %v7, ptr %d7, align 8
  ret void
}

define void @interleave_intrinsic_64_f32(<2 x float> %v0, <2 x float> %v1, <2 x float> %v2, <2 x float> %v3, <2 x float> %v4, <2 x float> %v5, <2 x float> %v6, <2 x float> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_f32:
  %wide = call <16 x float> @llvm.vector.interleave8.v16f32(<2 x float> %v0, <2 x float> %v1, <2 x float> %v2, <2 x float> %v3, <2 x float> %v4, <2 x float> %v5, <2 x float> %v6, <2 x float> %v7)
  store <16 x float> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_f32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_f32:
  %wide = load <16 x float>, ptr %src, align 8
  %parts = call { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.vector.deinterleave8.v16f32(<16 x float> %wide)
  %v0 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 0
  store <2 x float> %v0, ptr %dst, align 8
  %v1 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <2 x float> %v1, ptr %d1, align 8
  %v2 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <2 x float> %v2, ptr %d2, align 8
  %v3 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <2 x float> %v3, ptr %d3, align 8
  %v4 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <2 x float> %v4, ptr %d4, align 8
  %v5 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <2 x float> %v5, ptr %d5, align 8
  %v6 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 48
  store <2 x float> %v6, ptr %d6, align 8
  %v7 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 56
  store <2 x float> %v7, ptr %d7, align 8
  ret void
}

define void @interleave_intrinsic_128_i8(<16 x i8> %v0, <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3, <16 x i8> %v4, <16 x i8> %v5, <16 x i8> %v6, <16 x i8> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_i8:
  %wide = call <128 x i8> @llvm.vector.interleave8.v128i8(<16 x i8> %v0, <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3, <16 x i8> %v4, <16 x i8> %v5, <16 x i8> %v6, <16 x i8> %v7)
  store <128 x i8> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_i8(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_i8:
  %wide = load <128 x i8>, ptr %src, align 16
  %parts = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.vector.deinterleave8.v128i8(<128 x i8> %wide)
  %v0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 0
  store <16 x i8> %v0, ptr %dst, align 16
  %v1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <16 x i8> %v1, ptr %d1, align 16
  %v2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <16 x i8> %v2, ptr %d2, align 16
  %v3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <16 x i8> %v3, ptr %d3, align 16
  %v4 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <16 x i8> %v4, ptr %d4, align 16
  %v5 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <16 x i8> %v5, ptr %d5, align 16
  %v6 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 96
  store <16 x i8> %v6, ptr %d6, align 16
  %v7 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 112
  store <16 x i8> %v7, ptr %d7, align 16
  ret void
}

define void @interleave_intrinsic_128_i16(<8 x i16> %v0, <8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3, <8 x i16> %v4, <8 x i16> %v5, <8 x i16> %v6, <8 x i16> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_i16:
  %wide = call <64 x i16> @llvm.vector.interleave8.v64i16(<8 x i16> %v0, <8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3, <8 x i16> %v4, <8 x i16> %v5, <8 x i16> %v6, <8 x i16> %v7)
  store <64 x i16> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_i16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_i16:
  %wide = load <64 x i16>, ptr %src, align 16
  %parts = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.vector.deinterleave8.v64i16(<64 x i16> %wide)
  %v0 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 0
  store <8 x i16> %v0, ptr %dst, align 16
  %v1 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <8 x i16> %v1, ptr %d1, align 16
  %v2 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <8 x i16> %v2, ptr %d2, align 16
  %v3 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <8 x i16> %v3, ptr %d3, align 16
  %v4 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <8 x i16> %v4, ptr %d4, align 16
  %v5 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <8 x i16> %v5, ptr %d5, align 16
  %v6 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 96
  store <8 x i16> %v6, ptr %d6, align 16
  %v7 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 112
  store <8 x i16> %v7, ptr %d7, align 16
  ret void
}

define void @interleave_intrinsic_128_i32(<4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3, <4 x i32> %v4, <4 x i32> %v5, <4 x i32> %v6, <4 x i32> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_i32:
  %wide = call <32 x i32> @llvm.vector.interleave8.v32i32(<4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3, <4 x i32> %v4, <4 x i32> %v5, <4 x i32> %v6, <4 x i32> %v7)
  store <32 x i32> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_i32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_i32:
  %wide = load <32 x i32>, ptr %src, align 16
  %parts = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.vector.deinterleave8.v32i32(<32 x i32> %wide)
  %v0 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 0
  store <4 x i32> %v0, ptr %dst, align 16
  %v1 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <4 x i32> %v1, ptr %d1, align 16
  %v2 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <4 x i32> %v2, ptr %d2, align 16
  %v3 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <4 x i32> %v3, ptr %d3, align 16
  %v4 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <4 x i32> %v4, ptr %d4, align 16
  %v5 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <4 x i32> %v5, ptr %d5, align 16
  %v6 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 96
  store <4 x i32> %v6, ptr %d6, align 16
  %v7 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 112
  store <4 x i32> %v7, ptr %d7, align 16
  ret void
}

define void @interleave_intrinsic_128_i64(<2 x i64> %v0, <2 x i64> %v1, <2 x i64> %v2, <2 x i64> %v3, <2 x i64> %v4, <2 x i64> %v5, <2 x i64> %v6, <2 x i64> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_i64:
  %wide = call <16 x i64> @llvm.vector.interleave8.v16i64(<2 x i64> %v0, <2 x i64> %v1, <2 x i64> %v2, <2 x i64> %v3, <2 x i64> %v4, <2 x i64> %v5, <2 x i64> %v6, <2 x i64> %v7)
  store <16 x i64> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_i64(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_i64:
  %wide = load <16 x i64>, ptr %src, align 16
  %parts = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.vector.deinterleave8.v16i64(<16 x i64> %wide)
  %v0 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 0
  store <2 x i64> %v0, ptr %dst, align 16
  %v1 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <2 x i64> %v1, ptr %d1, align 16
  %v2 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <2 x i64> %v2, ptr %d2, align 16
  %v3 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <2 x i64> %v3, ptr %d3, align 16
  %v4 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <2 x i64> %v4, ptr %d4, align 16
  %v5 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <2 x i64> %v5, ptr %d5, align 16
  %v6 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 96
  store <2 x i64> %v6, ptr %d6, align 16
  %v7 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 112
  store <2 x i64> %v7, ptr %d7, align 16
  ret void
}

define void @interleave_intrinsic_128_f16(<8 x half> %v0, <8 x half> %v1, <8 x half> %v2, <8 x half> %v3, <8 x half> %v4, <8 x half> %v5, <8 x half> %v6, <8 x half> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_f16:
  %wide = call <64 x half> @llvm.vector.interleave8.v64f16(<8 x half> %v0, <8 x half> %v1, <8 x half> %v2, <8 x half> %v3, <8 x half> %v4, <8 x half> %v5, <8 x half> %v6, <8 x half> %v7)
  store <64 x half> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_f16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_f16:
  %wide = load <64 x half>, ptr %src, align 16
  %parts = call { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.vector.deinterleave8.v64f16(<64 x half> %wide)
  %v0 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 0
  store <8 x half> %v0, ptr %dst, align 16
  %v1 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <8 x half> %v1, ptr %d1, align 16
  %v2 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <8 x half> %v2, ptr %d2, align 16
  %v3 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <8 x half> %v3, ptr %d3, align 16
  %v4 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <8 x half> %v4, ptr %d4, align 16
  %v5 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <8 x half> %v5, ptr %d5, align 16
  %v6 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 96
  store <8 x half> %v6, ptr %d6, align 16
  %v7 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 112
  store <8 x half> %v7, ptr %d7, align 16
  ret void
}

define void @interleave_intrinsic_128_f32(<4 x float> %v0, <4 x float> %v1, <4 x float> %v2, <4 x float> %v3, <4 x float> %v4, <4 x float> %v5, <4 x float> %v6, <4 x float> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_f32:
  %wide = call <32 x float> @llvm.vector.interleave8.v32f32(<4 x float> %v0, <4 x float> %v1, <4 x float> %v2, <4 x float> %v3, <4 x float> %v4, <4 x float> %v5, <4 x float> %v6, <4 x float> %v7)
  store <32 x float> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_f32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_f32:
  %wide = load <32 x float>, ptr %src, align 16
  %parts = call { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.vector.deinterleave8.v32f32(<32 x float> %wide)
  %v0 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 0
  store <4 x float> %v0, ptr %dst, align 16
  %v1 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <4 x float> %v1, ptr %d1, align 16
  %v2 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <4 x float> %v2, ptr %d2, align 16
  %v3 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <4 x float> %v3, ptr %d3, align 16
  %v4 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <4 x float> %v4, ptr %d4, align 16
  %v5 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <4 x float> %v5, ptr %d5, align 16
  %v6 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 96
  store <4 x float> %v6, ptr %d6, align 16
  %v7 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 112
  store <4 x float> %v7, ptr %d7, align 16
  ret void
}

define void @interleave_intrinsic_128_f64(<2 x double> %v0, <2 x double> %v1, <2 x double> %v2, <2 x double> %v3, <2 x double> %v4, <2 x double> %v5, <2 x double> %v6, <2 x double> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_f64:
  %wide = call <16 x double> @llvm.vector.interleave8.v16f64(<2 x double> %v0, <2 x double> %v1, <2 x double> %v2, <2 x double> %v3, <2 x double> %v4, <2 x double> %v5, <2 x double> %v6, <2 x double> %v7)
  store <16 x double> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_f64(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_f64:
  %wide = load <16 x double>, ptr %src, align 16
  %parts = call { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.vector.deinterleave8.v16f64(<16 x double> %wide)
  %v0 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 0
  store <2 x double> %v0, ptr %dst, align 16
  %v1 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <2 x double> %v1, ptr %d1, align 16
  %v2 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <2 x double> %v2, ptr %d2, align 16
  %v3 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <2 x double> %v3, ptr %d3, align 16
  %v4 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <2 x double> %v4, ptr %d4, align 16
  %v5 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <2 x double> %v5, ptr %d5, align 16
  %v6 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 96
  store <2 x double> %v6, ptr %d6, align 16
  %v7 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 112
  store <2 x double> %v7, ptr %d7, align 16
  ret void
}

define void @interleave_intrinsic_256_i8(<32 x i8> %v0, <32 x i8> %v1, <32 x i8> %v2, <32 x i8> %v3, <32 x i8> %v4, <32 x i8> %v5, <32 x i8> %v6, <32 x i8> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_i8:
  %wide = call <256 x i8> @llvm.vector.interleave8.v256i8(<32 x i8> %v0, <32 x i8> %v1, <32 x i8> %v2, <32 x i8> %v3, <32 x i8> %v4, <32 x i8> %v5, <32 x i8> %v6, <32 x i8> %v7)
  store <256 x i8> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_i8(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_i8:
  %wide = load <256 x i8>, ptr %src, align 32
  %parts = call { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } @llvm.vector.deinterleave8.v256i8(<256 x i8> %wide)
  %v0 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 0
  store <32 x i8> %v0, ptr %dst, align 32
  %v1 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <32 x i8> %v1, ptr %d1, align 32
  %v2 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <32 x i8> %v2, ptr %d2, align 32
  %v3 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <32 x i8> %v3, ptr %d3, align 32
  %v4 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <32 x i8> %v4, ptr %d4, align 32
  %v5 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <32 x i8> %v5, ptr %d5, align 32
  %v6 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 192
  store <32 x i8> %v6, ptr %d6, align 32
  %v7 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 224
  store <32 x i8> %v7, ptr %d7, align 32
  ret void
}

define void @interleave_intrinsic_256_i16(<16 x i16> %v0, <16 x i16> %v1, <16 x i16> %v2, <16 x i16> %v3, <16 x i16> %v4, <16 x i16> %v5, <16 x i16> %v6, <16 x i16> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_i16:
  %wide = call <128 x i16> @llvm.vector.interleave8.v128i16(<16 x i16> %v0, <16 x i16> %v1, <16 x i16> %v2, <16 x i16> %v3, <16 x i16> %v4, <16 x i16> %v5, <16 x i16> %v6, <16 x i16> %v7)
  store <128 x i16> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_i16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_i16:
  %wide = load <128 x i16>, ptr %src, align 32
  %parts = call { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } @llvm.vector.deinterleave8.v128i16(<128 x i16> %wide)
  %v0 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 0
  store <16 x i16> %v0, ptr %dst, align 32
  %v1 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <16 x i16> %v1, ptr %d1, align 32
  %v2 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <16 x i16> %v2, ptr %d2, align 32
  %v3 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <16 x i16> %v3, ptr %d3, align 32
  %v4 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <16 x i16> %v4, ptr %d4, align 32
  %v5 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <16 x i16> %v5, ptr %d5, align 32
  %v6 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 192
  store <16 x i16> %v6, ptr %d6, align 32
  %v7 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 224
  store <16 x i16> %v7, ptr %d7, align 32
  ret void
}

define void @interleave_intrinsic_256_i32(<8 x i32> %v0, <8 x i32> %v1, <8 x i32> %v2, <8 x i32> %v3, <8 x i32> %v4, <8 x i32> %v5, <8 x i32> %v6, <8 x i32> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_i32:
  %wide = call <64 x i32> @llvm.vector.interleave8.v64i32(<8 x i32> %v0, <8 x i32> %v1, <8 x i32> %v2, <8 x i32> %v3, <8 x i32> %v4, <8 x i32> %v5, <8 x i32> %v6, <8 x i32> %v7)
  store <64 x i32> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_i32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_i32:
  %wide = load <64 x i32>, ptr %src, align 32
  %parts = call { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } @llvm.vector.deinterleave8.v64i32(<64 x i32> %wide)
  %v0 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 0
  store <8 x i32> %v0, ptr %dst, align 32
  %v1 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <8 x i32> %v1, ptr %d1, align 32
  %v2 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <8 x i32> %v2, ptr %d2, align 32
  %v3 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <8 x i32> %v3, ptr %d3, align 32
  %v4 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <8 x i32> %v4, ptr %d4, align 32
  %v5 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <8 x i32> %v5, ptr %d5, align 32
  %v6 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 192
  store <8 x i32> %v6, ptr %d6, align 32
  %v7 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 224
  store <8 x i32> %v7, ptr %d7, align 32
  ret void
}

define void @interleave_intrinsic_256_i64(<4 x i64> %v0, <4 x i64> %v1, <4 x i64> %v2, <4 x i64> %v3, <4 x i64> %v4, <4 x i64> %v5, <4 x i64> %v6, <4 x i64> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_i64:
  %wide = call <32 x i64> @llvm.vector.interleave8.v32i64(<4 x i64> %v0, <4 x i64> %v1, <4 x i64> %v2, <4 x i64> %v3, <4 x i64> %v4, <4 x i64> %v5, <4 x i64> %v6, <4 x i64> %v7)
  store <32 x i64> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_i64(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_i64:
  %wide = load <32 x i64>, ptr %src, align 32
  %parts = call { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } @llvm.vector.deinterleave8.v32i64(<32 x i64> %wide)
  %v0 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 0
  store <4 x i64> %v0, ptr %dst, align 32
  %v1 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <4 x i64> %v1, ptr %d1, align 32
  %v2 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <4 x i64> %v2, ptr %d2, align 32
  %v3 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <4 x i64> %v3, ptr %d3, align 32
  %v4 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <4 x i64> %v4, ptr %d4, align 32
  %v5 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <4 x i64> %v5, ptr %d5, align 32
  %v6 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 192
  store <4 x i64> %v6, ptr %d6, align 32
  %v7 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 224
  store <4 x i64> %v7, ptr %d7, align 32
  ret void
}

define void @interleave_intrinsic_256_f16(<16 x half> %v0, <16 x half> %v1, <16 x half> %v2, <16 x half> %v3, <16 x half> %v4, <16 x half> %v5, <16 x half> %v6, <16 x half> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_f16:
  %wide = call <128 x half> @llvm.vector.interleave8.v128f16(<16 x half> %v0, <16 x half> %v1, <16 x half> %v2, <16 x half> %v3, <16 x half> %v4, <16 x half> %v5, <16 x half> %v6, <16 x half> %v7)
  store <128 x half> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_f16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_f16:
  %wide = load <128 x half>, ptr %src, align 32
  %parts = call { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } @llvm.vector.deinterleave8.v128f16(<128 x half> %wide)
  %v0 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 0
  store <16 x half> %v0, ptr %dst, align 32
  %v1 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <16 x half> %v1, ptr %d1, align 32
  %v2 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <16 x half> %v2, ptr %d2, align 32
  %v3 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <16 x half> %v3, ptr %d3, align 32
  %v4 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <16 x half> %v4, ptr %d4, align 32
  %v5 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <16 x half> %v5, ptr %d5, align 32
  %v6 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 192
  store <16 x half> %v6, ptr %d6, align 32
  %v7 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 224
  store <16 x half> %v7, ptr %d7, align 32
  ret void
}

define void @interleave_intrinsic_256_f32(<8 x float> %v0, <8 x float> %v1, <8 x float> %v2, <8 x float> %v3, <8 x float> %v4, <8 x float> %v5, <8 x float> %v6, <8 x float> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_f32:
  %wide = call <64 x float> @llvm.vector.interleave8.v64f32(<8 x float> %v0, <8 x float> %v1, <8 x float> %v2, <8 x float> %v3, <8 x float> %v4, <8 x float> %v5, <8 x float> %v6, <8 x float> %v7)
  store <64 x float> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_f32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_f32:
  %wide = load <64 x float>, ptr %src, align 32
  %parts = call { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } @llvm.vector.deinterleave8.v64f32(<64 x float> %wide)
  %v0 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 0
  store <8 x float> %v0, ptr %dst, align 32
  %v1 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <8 x float> %v1, ptr %d1, align 32
  %v2 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <8 x float> %v2, ptr %d2, align 32
  %v3 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <8 x float> %v3, ptr %d3, align 32
  %v4 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <8 x float> %v4, ptr %d4, align 32
  %v5 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <8 x float> %v5, ptr %d5, align 32
  %v6 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 192
  store <8 x float> %v6, ptr %d6, align 32
  %v7 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 224
  store <8 x float> %v7, ptr %d7, align 32
  ret void
}

define void @interleave_intrinsic_256_f64(<4 x double> %v0, <4 x double> %v1, <4 x double> %v2, <4 x double> %v3, <4 x double> %v4, <4 x double> %v5, <4 x double> %v6, <4 x double> %v7, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_f64:
  %wide = call <32 x double> @llvm.vector.interleave8.v32f64(<4 x double> %v0, <4 x double> %v1, <4 x double> %v2, <4 x double> %v3, <4 x double> %v4, <4 x double> %v5, <4 x double> %v6, <4 x double> %v7)
  store <32 x double> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_f64(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_f64:
  %wide = load <32 x double>, ptr %src, align 32
  %parts = call { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } @llvm.vector.deinterleave8.v32f64(<32 x double> %wide)
  %v0 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 0
  store <4 x double> %v0, ptr %dst, align 32
  %v1 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <4 x double> %v1, ptr %d1, align 32
  %v2 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <4 x double> %v2, ptr %d2, align 32
  %v3 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <4 x double> %v3, ptr %d3, align 32
  %v4 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <4 x double> %v4, ptr %d4, align 32
  %v5 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <4 x double> %v5, ptr %d5, align 32
  %v6 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 6
  %d6 = getelementptr i8, ptr %dst, i64 192
  store <4 x double> %v6, ptr %d6, align 32
  %v7 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 7
  %d7 = getelementptr i8, ptr %dst, i64 224
  store <4 x double> %v7, ptr %d7, align 32
  ret void
}
