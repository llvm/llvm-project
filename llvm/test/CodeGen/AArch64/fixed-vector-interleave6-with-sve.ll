; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -force-streaming-compatible < %s | FileCheck %s --check-prefix=SVE

define void @interleave_intrinsic_64_i8(<8 x i8> %v0, <8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3, <8 x i8> %v4, <8 x i8> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_i8:
  %wide = call <48 x i8> @llvm.vector.interleave6.v48i8(<8 x i8> %v0, <8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3, <8 x i8> %v4, <8 x i8> %v5)
  store <48 x i8> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_i8(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_i8:
  %wide = load <48 x i8>, ptr %src, align 8
  %parts = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.vector.deinterleave6.v48i8(<48 x i8> %wide)
  %v0 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 0
  store <8 x i8> %v0, ptr %dst, align 8
  %v1 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <8 x i8> %v1, ptr %d1, align 8
  %v2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <8 x i8> %v2, ptr %d2, align 8
  %v3 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <8 x i8> %v3, ptr %d3, align 8
  %v4 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <8 x i8> %v4, ptr %d4, align 8
  %v5 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <8 x i8> %v5, ptr %d5, align 8
  ret void
}

define void @interleave_intrinsic_64_i16(<4 x i16> %v0, <4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3, <4 x i16> %v4, <4 x i16> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_i16:
  %wide = call <24 x i16> @llvm.vector.interleave6.v24i16(<4 x i16> %v0, <4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3, <4 x i16> %v4, <4 x i16> %v5)
  store <24 x i16> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_i16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_i16:
  %wide = load <24 x i16>, ptr %src, align 8
  %parts = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.vector.deinterleave6.v24i16(<24 x i16> %wide)
  %v0 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 0
  store <4 x i16> %v0, ptr %dst, align 8
  %v1 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <4 x i16> %v1, ptr %d1, align 8
  %v2 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <4 x i16> %v2, ptr %d2, align 8
  %v3 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <4 x i16> %v3, ptr %d3, align 8
  %v4 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <4 x i16> %v4, ptr %d4, align 8
  %v5 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <4 x i16> %v5, ptr %d5, align 8
  ret void
}

define void @interleave_intrinsic_64_i32(<2 x i32> %v0, <2 x i32> %v1, <2 x i32> %v2, <2 x i32> %v3, <2 x i32> %v4, <2 x i32> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_i32:
  %wide = call <12 x i32> @llvm.vector.interleave6.v12i32(<2 x i32> %v0, <2 x i32> %v1, <2 x i32> %v2, <2 x i32> %v3, <2 x i32> %v4, <2 x i32> %v5)
  store <12 x i32> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_i32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_i32:
  %wide = load <12 x i32>, ptr %src, align 8
  %parts = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.vector.deinterleave6.v12i32(<12 x i32> %wide)
  %v0 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 0
  store <2 x i32> %v0, ptr %dst, align 8
  %v1 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <2 x i32> %v1, ptr %d1, align 8
  %v2 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <2 x i32> %v2, ptr %d2, align 8
  %v3 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <2 x i32> %v3, ptr %d3, align 8
  %v4 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <2 x i32> %v4, ptr %d4, align 8
  %v5 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <2 x i32> %v5, ptr %d5, align 8
  ret void
}

define void @interleave_intrinsic_64_f16(<4 x half> %v0, <4 x half> %v1, <4 x half> %v2, <4 x half> %v3, <4 x half> %v4, <4 x half> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_f16:
  %wide = call <24 x half> @llvm.vector.interleave6.v24f16(<4 x half> %v0, <4 x half> %v1, <4 x half> %v2, <4 x half> %v3, <4 x half> %v4, <4 x half> %v5)
  store <24 x half> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_f16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_f16:
  %wide = load <24 x half>, ptr %src, align 8
  %parts = call { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.vector.deinterleave6.v24f16(<24 x half> %wide)
  %v0 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 0
  store <4 x half> %v0, ptr %dst, align 8
  %v1 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <4 x half> %v1, ptr %d1, align 8
  %v2 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <4 x half> %v2, ptr %d2, align 8
  %v3 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <4 x half> %v3, ptr %d3, align 8
  %v4 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <4 x half> %v4, ptr %d4, align 8
  %v5 = extractvalue { <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half>, <4 x half> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <4 x half> %v5, ptr %d5, align 8
  ret void
}

define void @interleave_intrinsic_64_f32(<2 x float> %v0, <2 x float> %v1, <2 x float> %v2, <2 x float> %v3, <2 x float> %v4, <2 x float> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_64_f32:
  %wide = call <12 x float> @llvm.vector.interleave6.v12f32(<2 x float> %v0, <2 x float> %v1, <2 x float> %v2, <2 x float> %v3, <2 x float> %v4, <2 x float> %v5)
  store <12 x float> %wide, ptr %dst, align 8
  ret void
}

define void @deinterleave_intrinsic_64_f32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_64_f32:
  %wide = load <12 x float>, ptr %src, align 8
  %parts = call { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.vector.deinterleave6.v12f32(<12 x float> %wide)
  %v0 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 0
  store <2 x float> %v0, ptr %dst, align 8
  %v1 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 8
  store <2 x float> %v1, ptr %d1, align 8
  %v2 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 16
  store <2 x float> %v2, ptr %d2, align 8
  %v3 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 24
  store <2 x float> %v3, ptr %d3, align 8
  %v4 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 32
  store <2 x float> %v4, ptr %d4, align 8
  %v5 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float>, <2 x float> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 40
  store <2 x float> %v5, ptr %d5, align 8
  ret void
}

define void @interleave_intrinsic_128_i8(<16 x i8> %v0, <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3, <16 x i8> %v4, <16 x i8> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_i8:
  %wide = call <96 x i8> @llvm.vector.interleave6.v96i8(<16 x i8> %v0, <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3, <16 x i8> %v4, <16 x i8> %v5)
  store <96 x i8> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_i8(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_i8:
  %wide = load <96 x i8>, ptr %src, align 16
  %parts = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.vector.deinterleave6.v96i8(<96 x i8> %wide)
  %v0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 0
  store <16 x i8> %v0, ptr %dst, align 16
  %v1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <16 x i8> %v1, ptr %d1, align 16
  %v2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <16 x i8> %v2, ptr %d2, align 16
  %v3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <16 x i8> %v3, ptr %d3, align 16
  %v4 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <16 x i8> %v4, ptr %d4, align 16
  %v5 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <16 x i8> %v5, ptr %d5, align 16
  ret void
}

define void @interleave_intrinsic_128_i16(<8 x i16> %v0, <8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3, <8 x i16> %v4, <8 x i16> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_i16:
  %wide = call <48 x i16> @llvm.vector.interleave6.v48i16(<8 x i16> %v0, <8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3, <8 x i16> %v4, <8 x i16> %v5)
  store <48 x i16> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_i16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_i16:
  %wide = load <48 x i16>, ptr %src, align 16
  %parts = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.vector.deinterleave6.v48i16(<48 x i16> %wide)
  %v0 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 0
  store <8 x i16> %v0, ptr %dst, align 16
  %v1 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <8 x i16> %v1, ptr %d1, align 16
  %v2 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <8 x i16> %v2, ptr %d2, align 16
  %v3 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <8 x i16> %v3, ptr %d3, align 16
  %v4 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <8 x i16> %v4, ptr %d4, align 16
  %v5 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <8 x i16> %v5, ptr %d5, align 16
  ret void
}

define void @interleave_intrinsic_128_i32(<4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3, <4 x i32> %v4, <4 x i32> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_i32:
  %wide = call <24 x i32> @llvm.vector.interleave6.v24i32(<4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3, <4 x i32> %v4, <4 x i32> %v5)
  store <24 x i32> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_i32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_i32:
  %wide = load <24 x i32>, ptr %src, align 16
  %parts = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.vector.deinterleave6.v24i32(<24 x i32> %wide)
  %v0 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 0
  store <4 x i32> %v0, ptr %dst, align 16
  %v1 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <4 x i32> %v1, ptr %d1, align 16
  %v2 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <4 x i32> %v2, ptr %d2, align 16
  %v3 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <4 x i32> %v3, ptr %d3, align 16
  %v4 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <4 x i32> %v4, ptr %d4, align 16
  %v5 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <4 x i32> %v5, ptr %d5, align 16
  ret void
}

define void @interleave_intrinsic_128_i64(<2 x i64> %v0, <2 x i64> %v1, <2 x i64> %v2, <2 x i64> %v3, <2 x i64> %v4, <2 x i64> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_i64:
  %wide = call <12 x i64> @llvm.vector.interleave6.v12i64(<2 x i64> %v0, <2 x i64> %v1, <2 x i64> %v2, <2 x i64> %v3, <2 x i64> %v4, <2 x i64> %v5)
  store <12 x i64> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_i64(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_i64:
  %wide = load <12 x i64>, ptr %src, align 16
  %parts = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.vector.deinterleave6.v12i64(<12 x i64> %wide)
  %v0 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 0
  store <2 x i64> %v0, ptr %dst, align 16
  %v1 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <2 x i64> %v1, ptr %d1, align 16
  %v2 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <2 x i64> %v2, ptr %d2, align 16
  %v3 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <2 x i64> %v3, ptr %d3, align 16
  %v4 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <2 x i64> %v4, ptr %d4, align 16
  %v5 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <2 x i64> %v5, ptr %d5, align 16
  ret void
}

define void @interleave_intrinsic_128_f16(<8 x half> %v0, <8 x half> %v1, <8 x half> %v2, <8 x half> %v3, <8 x half> %v4, <8 x half> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_f16:
  %wide = call <48 x half> @llvm.vector.interleave6.v48f16(<8 x half> %v0, <8 x half> %v1, <8 x half> %v2, <8 x half> %v3, <8 x half> %v4, <8 x half> %v5)
  store <48 x half> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_f16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_f16:
  %wide = load <48 x half>, ptr %src, align 16
  %parts = call { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.vector.deinterleave6.v48f16(<48 x half> %wide)
  %v0 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 0
  store <8 x half> %v0, ptr %dst, align 16
  %v1 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <8 x half> %v1, ptr %d1, align 16
  %v2 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <8 x half> %v2, ptr %d2, align 16
  %v3 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <8 x half> %v3, ptr %d3, align 16
  %v4 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <8 x half> %v4, ptr %d4, align 16
  %v5 = extractvalue { <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half>, <8 x half> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <8 x half> %v5, ptr %d5, align 16
  ret void
}

define void @interleave_intrinsic_128_f32(<4 x float> %v0, <4 x float> %v1, <4 x float> %v2, <4 x float> %v3, <4 x float> %v4, <4 x float> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_f32:
  %wide = call <24 x float> @llvm.vector.interleave6.v24f32(<4 x float> %v0, <4 x float> %v1, <4 x float> %v2, <4 x float> %v3, <4 x float> %v4, <4 x float> %v5)
  store <24 x float> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_f32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_f32:
  %wide = load <24 x float>, ptr %src, align 16
  %parts = call { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.vector.deinterleave6.v24f32(<24 x float> %wide)
  %v0 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 0
  store <4 x float> %v0, ptr %dst, align 16
  %v1 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <4 x float> %v1, ptr %d1, align 16
  %v2 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <4 x float> %v2, ptr %d2, align 16
  %v3 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <4 x float> %v3, ptr %d3, align 16
  %v4 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <4 x float> %v4, ptr %d4, align 16
  %v5 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <4 x float> %v5, ptr %d5, align 16
  ret void
}

define void @interleave_intrinsic_128_f64(<2 x double> %v0, <2 x double> %v1, <2 x double> %v2, <2 x double> %v3, <2 x double> %v4, <2 x double> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_128_f64:
  %wide = call <12 x double> @llvm.vector.interleave6.v12f64(<2 x double> %v0, <2 x double> %v1, <2 x double> %v2, <2 x double> %v3, <2 x double> %v4, <2 x double> %v5)
  store <12 x double> %wide, ptr %dst, align 16
  ret void
}

define void @deinterleave_intrinsic_128_f64(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_128_f64:
  %wide = load <12 x double>, ptr %src, align 16
  %parts = call { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.vector.deinterleave6.v12f64(<12 x double> %wide)
  %v0 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 0
  store <2 x double> %v0, ptr %dst, align 16
  %v1 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 16
  store <2 x double> %v1, ptr %d1, align 16
  %v2 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 32
  store <2 x double> %v2, ptr %d2, align 16
  %v3 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 48
  store <2 x double> %v3, ptr %d3, align 16
  %v4 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 64
  store <2 x double> %v4, ptr %d4, align 16
  %v5 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double>, <2 x double> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 80
  store <2 x double> %v5, ptr %d5, align 16
  ret void
}

define void @interleave_intrinsic_256_i8(<32 x i8> %v0, <32 x i8> %v1, <32 x i8> %v2, <32 x i8> %v3, <32 x i8> %v4, <32 x i8> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_i8:
  %wide = call <192 x i8> @llvm.vector.interleave6.v192i8(<32 x i8> %v0, <32 x i8> %v1, <32 x i8> %v2, <32 x i8> %v3, <32 x i8> %v4, <32 x i8> %v5)
  store <192 x i8> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_i8(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_i8:
  %wide = load <192 x i8>, ptr %src, align 32
  %parts = call { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } @llvm.vector.deinterleave6.v192i8(<192 x i8> %wide)
  %v0 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 0
  store <32 x i8> %v0, ptr %dst, align 32
  %v1 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <32 x i8> %v1, ptr %d1, align 32
  %v2 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <32 x i8> %v2, ptr %d2, align 32
  %v3 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <32 x i8> %v3, ptr %d3, align 32
  %v4 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <32 x i8> %v4, ptr %d4, align 32
  %v5 = extractvalue { <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8>, <32 x i8> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <32 x i8> %v5, ptr %d5, align 32
  ret void
}

define void @interleave_intrinsic_256_i16(<16 x i16> %v0, <16 x i16> %v1, <16 x i16> %v2, <16 x i16> %v3, <16 x i16> %v4, <16 x i16> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_i16:
  %wide = call <96 x i16> @llvm.vector.interleave6.v96i16(<16 x i16> %v0, <16 x i16> %v1, <16 x i16> %v2, <16 x i16> %v3, <16 x i16> %v4, <16 x i16> %v5)
  store <96 x i16> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_i16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_i16:
  %wide = load <96 x i16>, ptr %src, align 32
  %parts = call { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } @llvm.vector.deinterleave6.v96i16(<96 x i16> %wide)
  %v0 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 0
  store <16 x i16> %v0, ptr %dst, align 32
  %v1 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <16 x i16> %v1, ptr %d1, align 32
  %v2 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <16 x i16> %v2, ptr %d2, align 32
  %v3 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <16 x i16> %v3, ptr %d3, align 32
  %v4 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <16 x i16> %v4, ptr %d4, align 32
  %v5 = extractvalue { <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16>, <16 x i16> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <16 x i16> %v5, ptr %d5, align 32
  ret void
}

define void @interleave_intrinsic_256_i32(<8 x i32> %v0, <8 x i32> %v1, <8 x i32> %v2, <8 x i32> %v3, <8 x i32> %v4, <8 x i32> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_i32:
  %wide = call <48 x i32> @llvm.vector.interleave6.v48i32(<8 x i32> %v0, <8 x i32> %v1, <8 x i32> %v2, <8 x i32> %v3, <8 x i32> %v4, <8 x i32> %v5)
  store <48 x i32> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_i32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_i32:
  %wide = load <48 x i32>, ptr %src, align 32
  %parts = call { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } @llvm.vector.deinterleave6.v48i32(<48 x i32> %wide)
  %v0 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 0
  store <8 x i32> %v0, ptr %dst, align 32
  %v1 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <8 x i32> %v1, ptr %d1, align 32
  %v2 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <8 x i32> %v2, ptr %d2, align 32
  %v3 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <8 x i32> %v3, ptr %d3, align 32
  %v4 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <8 x i32> %v4, ptr %d4, align 32
  %v5 = extractvalue { <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <8 x i32> %v5, ptr %d5, align 32
  ret void
}

define void @interleave_intrinsic_256_i64(<4 x i64> %v0, <4 x i64> %v1, <4 x i64> %v2, <4 x i64> %v3, <4 x i64> %v4, <4 x i64> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_i64:
  %wide = call <24 x i64> @llvm.vector.interleave6.v24i64(<4 x i64> %v0, <4 x i64> %v1, <4 x i64> %v2, <4 x i64> %v3, <4 x i64> %v4, <4 x i64> %v5)
  store <24 x i64> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_i64(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_i64:
  %wide = load <24 x i64>, ptr %src, align 32
  %parts = call { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } @llvm.vector.deinterleave6.v24i64(<24 x i64> %wide)
  %v0 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 0
  store <4 x i64> %v0, ptr %dst, align 32
  %v1 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <4 x i64> %v1, ptr %d1, align 32
  %v2 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <4 x i64> %v2, ptr %d2, align 32
  %v3 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <4 x i64> %v3, ptr %d3, align 32
  %v4 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <4 x i64> %v4, ptr %d4, align 32
  %v5 = extractvalue { <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, <4 x i64> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <4 x i64> %v5, ptr %d5, align 32
  ret void
}

define void @interleave_intrinsic_256_f16(<16 x half> %v0, <16 x half> %v1, <16 x half> %v2, <16 x half> %v3, <16 x half> %v4, <16 x half> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_f16:
  %wide = call <96 x half> @llvm.vector.interleave6.v96f16(<16 x half> %v0, <16 x half> %v1, <16 x half> %v2, <16 x half> %v3, <16 x half> %v4, <16 x half> %v5)
  store <96 x half> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_f16(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_f16:
  %wide = load <96 x half>, ptr %src, align 32
  %parts = call { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } @llvm.vector.deinterleave6.v96f16(<96 x half> %wide)
  %v0 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 0
  store <16 x half> %v0, ptr %dst, align 32
  %v1 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <16 x half> %v1, ptr %d1, align 32
  %v2 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <16 x half> %v2, ptr %d2, align 32
  %v3 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <16 x half> %v3, ptr %d3, align 32
  %v4 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <16 x half> %v4, ptr %d4, align 32
  %v5 = extractvalue { <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half>, <16 x half> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <16 x half> %v5, ptr %d5, align 32
  ret void
}

define void @interleave_intrinsic_256_f32(<8 x float> %v0, <8 x float> %v1, <8 x float> %v2, <8 x float> %v3, <8 x float> %v4, <8 x float> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_f32:
  %wide = call <48 x float> @llvm.vector.interleave6.v48f32(<8 x float> %v0, <8 x float> %v1, <8 x float> %v2, <8 x float> %v3, <8 x float> %v4, <8 x float> %v5)
  store <48 x float> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_f32(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_f32:
  %wide = load <48 x float>, ptr %src, align 32
  %parts = call { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } @llvm.vector.deinterleave6.v48f32(<48 x float> %wide)
  %v0 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 0
  store <8 x float> %v0, ptr %dst, align 32
  %v1 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <8 x float> %v1, ptr %d1, align 32
  %v2 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <8 x float> %v2, ptr %d2, align 32
  %v3 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <8 x float> %v3, ptr %d3, align 32
  %v4 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <8 x float> %v4, ptr %d4, align 32
  %v5 = extractvalue { <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float>, <8 x float> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <8 x float> %v5, ptr %d5, align 32
  ret void
}

define void @interleave_intrinsic_256_f64(<4 x double> %v0, <4 x double> %v1, <4 x double> %v2, <4 x double> %v3, <4 x double> %v4, <4 x double> %v5, ptr noalias %dst) {
; SVE-LABEL: interleave_intrinsic_256_f64:
  %wide = call <24 x double> @llvm.vector.interleave6.v24f64(<4 x double> %v0, <4 x double> %v1, <4 x double> %v2, <4 x double> %v3, <4 x double> %v4, <4 x double> %v5)
  store <24 x double> %wide, ptr %dst, align 32
  ret void
}

define void @deinterleave_intrinsic_256_f64(ptr noalias %src, ptr noalias %dst) {
; SVE-LABEL: deinterleave_intrinsic_256_f64:
  %wide = load <24 x double>, ptr %src, align 32
  %parts = call { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } @llvm.vector.deinterleave6.v24f64(<24 x double> %wide)
  %v0 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 0
  store <4 x double> %v0, ptr %dst, align 32
  %v1 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 1
  %d1 = getelementptr i8, ptr %dst, i64 32
  store <4 x double> %v1, ptr %d1, align 32
  %v2 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 2
  %d2 = getelementptr i8, ptr %dst, i64 64
  store <4 x double> %v2, ptr %d2, align 32
  %v3 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 3
  %d3 = getelementptr i8, ptr %dst, i64 96
  store <4 x double> %v3, ptr %d3, align 32
  %v4 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 4
  %d4 = getelementptr i8, ptr %dst, i64 128
  store <4 x double> %v4, ptr %d4, align 32
  %v5 = extractvalue { <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double> } %parts, 5
  %d5 = getelementptr i8, ptr %dst, i64 160
  store <4 x double> %v5, ptr %d5, align 32
  ret void
}
