; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define void @vst1i8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst1i8:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vst1.8 {d16}, [r0, :64]
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst1.v8i8(i8* %A, <8 x i8> %tmp1, i32 16)
	ret void
}

define void @vst1i16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst1i16:
;CHECK: vst1.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst1.v4i16(i8* %tmp0, <4 x i16> %tmp1, i32 1)
	ret void
}

define void @vst1i32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst1i32:
;CHECK: vst1.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst1.v2i32(i8* %tmp0, <2 x i32> %tmp1, i32 1)
	ret void
}

define void @vst1f(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst1f:
;CHECK: vst1.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst1.v2f32(i8* %tmp0, <2 x float> %tmp1, i32 1)
	ret void
}

;Check for a post-increment updating store.
define void @vst1f_update(float** %ptr, <2 x float>* %B) nounwind {
;CHECK: vst1f_update:
;CHECK: vst1.32 {d16}, [r1]!
	%A = load float** %ptr
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst1.v2f32(i8* %tmp0, <2 x float> %tmp1, i32 1)
	%tmp2 = getelementptr float* %A, i32 2
	store float* %tmp2, float** %ptr
	ret void
}

define void @vst1i64(i64* %A, <1 x i64>* %B) nounwind {
;CHECK: vst1i64:
;CHECK: vst1.64
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = load <1 x i64>* %B
	call void @llvm.arm.neon.vst1.v1i64(i8* %tmp0, <1 x i64> %tmp1, i32 1)
	ret void
}

define void @vst1Qi8(i8* %A, <16 x i8>* %B) nounwind {
;CHECK: vst1Qi8:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vst1.8 {d16, d17}, [r0, :64]
	%tmp1 = load <16 x i8>* %B
	call void @llvm.arm.neon.vst1.v16i8(i8* %A, <16 x i8> %tmp1, i32 8)
	ret void
}

define void @vst1Qi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst1Qi16:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vst1.16 {d16, d17}, [r0, :128]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst1.v8i16(i8* %tmp0, <8 x i16> %tmp1, i32 32)
	ret void
}

;Check for a post-increment updating store with register increment.
define void @vst1Qi16_update(i16** %ptr, <8 x i16>* %B, i32 %inc) nounwind {
;CHECK: vst1Qi16_update:
;CHECK: vst1.16 {d16, d17}, [r1, :64], r2
	%A = load i16** %ptr
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst1.v8i16(i8* %tmp0, <8 x i16> %tmp1, i32 8)
	%tmp2 = getelementptr i16* %A, i32 %inc
	store i16* %tmp2, i16** %ptr
	ret void
}

define void @vst1Qi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst1Qi32:
;CHECK: vst1.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst1.v4i32(i8* %tmp0, <4 x i32> %tmp1, i32 1)
	ret void
}

define void @vst1Qf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst1Qf:
;CHECK: vst1.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vst1.v4f32(i8* %tmp0, <4 x float> %tmp1, i32 1)
	ret void
}

define void @vst1Qi64(i64* %A, <2 x i64>* %B) nounwind {
;CHECK: vst1Qi64:
;CHECK: vst1.64
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = load <2 x i64>* %B
	call void @llvm.arm.neon.vst1.v2i64(i8* %tmp0, <2 x i64> %tmp1, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst1.v8i8(i8*, <8 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst1.v4i16(i8*, <4 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst1.v2i32(i8*, <2 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst1.v2f32(i8*, <2 x float>, i32) nounwind
declare void @llvm.arm.neon.vst1.v1i64(i8*, <1 x i64>, i32) nounwind

declare void @llvm.arm.neon.vst1.v16i8(i8*, <16 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst1.v8i16(i8*, <8 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst1.v4i32(i8*, <4 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32) nounwind
declare void @llvm.arm.neon.vst1.v2i64(i8*, <2 x i64>, i32) nounwind
