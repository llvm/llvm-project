; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s
; RUN: llc < %s -march=arm -mattr=+neon -regalloc=basic | FileCheck %s

define <8 x i8> @vld1i8(i8* %A) nounwind {
;CHECK: vld1i8:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld1.8 {d16}, [r0, :64]
	%tmp1 = call <8 x i8> @llvm.arm.neon.vld1.v8i8(i8* %A, i32 16)
	ret <8 x i8> %tmp1
}

define <4 x i16> @vld1i16(i16* %A) nounwind {
;CHECK: vld1i16:
;CHECK: vld1.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call <4 x i16> @llvm.arm.neon.vld1.v4i16(i8* %tmp0, i32 1)
	ret <4 x i16> %tmp1
}

;Check for a post-increment updating load. 
define <4 x i16> @vld1i16_update(i16** %ptr) nounwind {
;CHECK: vld1i16_update:
;CHECK: vld1.16 {d16}, [{{r[0-9]+}}]!
	%A = load i16** %ptr
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call <4 x i16> @llvm.arm.neon.vld1.v4i16(i8* %tmp0, i32 1)
	%tmp2 = getelementptr i16* %A, i32 4
	       store i16* %tmp2, i16** %ptr
	ret <4 x i16> %tmp1
}

define <2 x i32> @vld1i32(i32* %A) nounwind {
;CHECK: vld1i32:
;CHECK: vld1.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call <2 x i32> @llvm.arm.neon.vld1.v2i32(i8* %tmp0, i32 1)
	ret <2 x i32> %tmp1
}

;Check for a post-increment updating load with register increment.
define <2 x i32> @vld1i32_update(i32** %ptr, i32 %inc) nounwind {
;CHECK: vld1i32_update:
;CHECK: vld1.32 {d16}, [{{r[0-9]+}}], {{r[0-9]+}}
	%A = load i32** %ptr
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call <2 x i32> @llvm.arm.neon.vld1.v2i32(i8* %tmp0, i32 1)
	%tmp2 = getelementptr i32* %A, i32 %inc
	store i32* %tmp2, i32** %ptr
	ret <2 x i32> %tmp1
}

define <2 x float> @vld1f(float* %A) nounwind {
;CHECK: vld1f:
;CHECK: vld1.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call <2 x float> @llvm.arm.neon.vld1.v2f32(i8* %tmp0, i32 1)
	ret <2 x float> %tmp1
}

define <1 x i64> @vld1i64(i64* %A) nounwind {
;CHECK: vld1i64:
;CHECK: vld1.64
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %tmp0, i32 1)
	ret <1 x i64> %tmp1
}

define <16 x i8> @vld1Qi8(i8* %A) nounwind {
;CHECK: vld1Qi8:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vld1.8 {d16, d17}, [r0, :64]
	%tmp1 = call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %A, i32 8)
	ret <16 x i8> %tmp1
}

;Check for a post-increment updating load.
define <16 x i8> @vld1Qi8_update(i8** %ptr) nounwind {
;CHECK: vld1Qi8_update:
;CHECK: vld1.8 {d16, d17}, [{{r[0-9]+}}, :64]!
	%A = load i8** %ptr
	%tmp1 = call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %A, i32 8)
	%tmp2 = getelementptr i8* %A, i32 16
	store i8* %tmp2, i8** %ptr
	ret <16 x i8> %tmp1
}

define <8 x i16> @vld1Qi16(i16* %A) nounwind {
;CHECK: vld1Qi16:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vld1.16 {d16, d17}, [r0, :128]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call <8 x i16> @llvm.arm.neon.vld1.v8i16(i8* %tmp0, i32 32)
	ret <8 x i16> %tmp1
}

define <4 x i32> @vld1Qi32(i32* %A) nounwind {
;CHECK: vld1Qi32:
;CHECK: vld1.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call <4 x i32> @llvm.arm.neon.vld1.v4i32(i8* %tmp0, i32 1)
	ret <4 x i32> %tmp1
}

define <4 x float> @vld1Qf(float* %A) nounwind {
;CHECK: vld1Qf:
;CHECK: vld1.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* %tmp0, i32 1)
	ret <4 x float> %tmp1
}

define <2 x i64> @vld1Qi64(i64* %A) nounwind {
;CHECK: vld1Qi64:
;CHECK: vld1.64
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = call <2 x i64> @llvm.arm.neon.vld1.v2i64(i8* %tmp0, i32 1)
	ret <2 x i64> %tmp1
}

declare <8 x i8>  @llvm.arm.neon.vld1.v8i8(i8*, i32) nounwind readonly
declare <4 x i16> @llvm.arm.neon.vld1.v4i16(i8*, i32) nounwind readonly
declare <2 x i32> @llvm.arm.neon.vld1.v2i32(i8*, i32) nounwind readonly
declare <2 x float> @llvm.arm.neon.vld1.v2f32(i8*, i32) nounwind readonly
declare <1 x i64> @llvm.arm.neon.vld1.v1i64(i8*, i32) nounwind readonly

declare <16 x i8> @llvm.arm.neon.vld1.v16i8(i8*, i32) nounwind readonly
declare <8 x i16> @llvm.arm.neon.vld1.v8i16(i8*, i32) nounwind readonly
declare <4 x i32> @llvm.arm.neon.vld1.v4i32(i8*, i32) nounwind readonly
declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*, i32) nounwind readonly
declare <2 x i64> @llvm.arm.neon.vld1.v2i64(i8*, i32) nounwind readonly

; Radar 8355607
; Do not crash if the vld1 result is not used.
define void @unused_vld1_result() {
entry:
  %0 = call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* undef, i32 1) 
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap() nounwind
