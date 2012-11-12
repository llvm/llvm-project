; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define void @vst1lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst1lanei8:
;Check the (default) alignment.
;CHECK: vst1.8 {d16[3]}, [r0]
	%tmp1 = load <8 x i8>* %B
        %tmp2 = extractelement <8 x i8> %tmp1, i32 3
        store i8 %tmp2, i8* %A, align 8
	ret void
}

;Check for a post-increment updating store.
define void @vst1lanei8_update(i8** %ptr, <8 x i8>* %B) nounwind {
;CHECK: vst1lanei8_update:
;CHECK: vst1.8 {d16[3]}, [r2]!
	%A = load i8** %ptr
	%tmp1 = load <8 x i8>* %B
	%tmp2 = extractelement <8 x i8> %tmp1, i32 3
	store i8 %tmp2, i8* %A, align 8
	%tmp3 = getelementptr i8* %A, i32 1
	store i8* %tmp3, i8** %ptr
	ret void
}

define void @vst1lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst1lanei16:
;Check the alignment value.  Max for this instruction is 16 bits:
;CHECK: vst1.16 {d16[2]}, [r0, :16]
	%tmp1 = load <4 x i16>* %B
        %tmp2 = extractelement <4 x i16> %tmp1, i32 2
        store i16 %tmp2, i16* %A, align 8
	ret void
}

define void @vst1lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst1lanei32:
;Check the alignment value.  Max for this instruction is 32 bits:
;CHECK: vst1.32 {d16[1]}, [r0, :32]
	%tmp1 = load <2 x i32>* %B
        %tmp2 = extractelement <2 x i32> %tmp1, i32 1
        store i32 %tmp2, i32* %A, align 8
	ret void
}

define void @vst1lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst1lanef:
;CHECK: vst1.32 {d16[1]}, [r0, :32]
	%tmp1 = load <2 x float>* %B
        %tmp2 = extractelement <2 x float> %tmp1, i32 1
        store float %tmp2, float* %A
	ret void
}

define void @vst1laneQi8(i8* %A, <16 x i8>* %B) nounwind {
;CHECK: vst1laneQi8:
; // Can use scalar load. No need to use vectors.
; // CHE-CK: vst1.8 {d17[1]}, [r0]
	%tmp1 = load <16 x i8>* %B
        %tmp2 = extractelement <16 x i8> %tmp1, i32 9
        store i8 %tmp2, i8* %A, align 8
	ret void
}

define void @vst1laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst1laneQi16:
;CHECK: vst1.16 {d17[1]}, [r0, :16]
	%tmp1 = load <8 x i16>* %B
        %tmp2 = extractelement <8 x i16> %tmp1, i32 5
        store i16 %tmp2, i16* %A, align 8
	ret void
}

define void @vst1laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst1laneQi32:
; // Can use scalar load. No need to use vectors.
; // CHE-CK: vst1.32 {d17[1]}, [r0, :32]
	%tmp1 = load <4 x i32>* %B
        %tmp2 = extractelement <4 x i32> %tmp1, i32 3
        store i32 %tmp2, i32* %A, align 8
	ret void
}

;Check for a post-increment updating store.
define void @vst1laneQi32_update(i32** %ptr, <4 x i32>* %B) nounwind {
;CHECK: vst1laneQi32_update:
; // Can use scalar load. No need to use vectors.
; // CHE-CK: vst1.32 {d17[1]}, [r1, :32]!
	%A = load i32** %ptr
	%tmp1 = load <4 x i32>* %B
	%tmp2 = extractelement <4 x i32> %tmp1, i32 3
	store i32 %tmp2, i32* %A, align 8
	%tmp3 = getelementptr i32* %A, i32 1
	store i32* %tmp3, i32** %ptr
	ret void
}

define void @vst1laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst1laneQf:
; // Can use scalar load. No need to use vectors.
; // CHE-CK: vst1.32 {d17[1]}, [r0]
	%tmp1 = load <4 x float>* %B
        %tmp2 = extractelement <4 x float> %tmp1, i32 3
        store float %tmp2, float* %A
	ret void
}

define void @vst2lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst2lanei8:
;Check the alignment value.  Max for this instruction is 16 bits:
;CHECK: vst2.8 {d16[1], d17[1]}, [r0, :16]
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst2lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 4)
	ret void
}

define void @vst2lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst2lanei16:
;Check the alignment value.  Max for this instruction is 32 bits:
;CHECK: vst2.16 {d16[1], d17[1]}, [r0, :32]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst2lane.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 8)
	ret void
}

;Check for a post-increment updating store with register increment.
define void @vst2lanei16_update(i16** %ptr, <4 x i16>* %B, i32 %inc) nounwind {
;CHECK: vst2lanei16_update:
;CHECK: vst2.16 {d16[1], d17[1]}, [r1], r2
	%A = load i16** %ptr
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst2lane.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 2)
	%tmp2 = getelementptr i16* %A, i32 %inc
	store i16* %tmp2, i16** %ptr
	ret void
}

define void @vst2lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst2lanei32:
;CHECK: vst2.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst2lane.v2i32(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst2lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst2lanef:
;CHECK: vst2.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst2lane.v2f32(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst2laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst2laneQi16:
;Check the (default) alignment.
;CHECK: vst2.16 {d17[1], d19[1]}, [r0]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst2lane.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 5, i32 1)
	ret void
}

define void @vst2laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst2laneQi32:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vst2.32 {d17[0], d19[0]}, [r0, :64]
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst2lane.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2, i32 16)
	ret void
}

define void @vst2laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst2laneQf:
;CHECK: vst2.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vst2lane.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, i32 3, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v4i16(i8*, <4 x i16>, <4 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v2i32(i8*, <2 x i32>, <2 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v2f32(i8*, <2 x float>, <2 x float>, i32, i32) nounwind

declare void @llvm.arm.neon.vst2lane.v8i16(i8*, <8 x i16>, <8 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v4i32(i8*, <4 x i32>, <4 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v4f32(i8*, <4 x float>, <4 x float>, i32, i32) nounwind

define void @vst3lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst3lanei8:
;CHECK: vst3.8
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst3lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst3lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst3lanei16:
;Check the (default) alignment value.  VST3 does not support alignment.
;CHECK: vst3.16 {d16[1], d17[1], d18[1]}, [r0]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst3lane.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 8)
	ret void
}

define void @vst3lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst3lanei32:
;CHECK: vst3.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst3lane.v2i32(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst3lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst3lanef:
;CHECK: vst3.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst3lane.v2f32(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst3laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst3laneQi16:
;Check the (default) alignment value.  VST3 does not support alignment.
;CHECK: vst3.16 {d17[2], d19[2], d21[2]}, [r0]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst3lane.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 6, i32 8)
	ret void
}

define void @vst3laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst3laneQi32:
;CHECK: vst3.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst3lane.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 0, i32 1)
	ret void
}

;Check for a post-increment updating store.
define void @vst3laneQi32_update(i32** %ptr, <4 x i32>* %B) nounwind {
;CHECK: vst3laneQi32_update:
;CHECK: vst3.32 {d16[0], d18[0], d20[0]}, [r1]!
	%A = load i32** %ptr
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst3lane.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 0, i32 1)
	%tmp2 = getelementptr i32* %A, i32 3
	store i32* %tmp2, i32** %ptr
	ret void
}

define void @vst3laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst3laneQf:
;CHECK: vst3.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vst3lane.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst3lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32, i32) nounwind

declare void @llvm.arm.neon.vst3lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, i32, i32) nounwind


define void @vst4lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst4lanei8:
;Check the alignment value.  Max for this instruction is 32 bits:
;CHECK: vst4.8 {d16[1], d17[1], d18[1], d19[1]}, [r0, :32]
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst4lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 8)
	ret void
}

;Check for a post-increment updating store.
define void @vst4lanei8_update(i8** %ptr, <8 x i8>* %B) nounwind {
;CHECK: vst4lanei8_update:
;CHECK: vst4.8 {d16[1], d17[1], d18[1], d19[1]}, [r1, :32]!
	%A = load i8** %ptr
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst4lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 8)
	%tmp2 = getelementptr i8* %A, i32 4
	store i8* %tmp2, i8** %ptr
	ret void
}

define void @vst4lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst4lanei16:
;CHECK: vst4.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst4lane.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst4lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst4lanei32:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vst4.32 {d16[1], d17[1], d18[1], d19[1]}, [r0, :128]
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst4lane.v2i32(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 16)
	ret void
}

define void @vst4lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst4lanef:
;CHECK: vst4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst4lane.v2f32(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst4laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst4laneQi16:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vst4.16 {d17[3], d19[3], d21[3], d23[3]}, [r0, :64]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst4lane.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 7, i32 16)
	ret void
}

define void @vst4laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst4laneQi32:
;Check the (default) alignment.
;CHECK: vst4.32 {d17[0], d19[0], d21[0], d23[0]}, [r0]
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst4lane.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2, i32 1)
	ret void
}

define void @vst4laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst4laneQf:
;CHECK: vst4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vst4lane.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
	ret void
}

; Make sure this doesn't crash; PR10258
define <8 x i16> @variable_insertelement(<8 x i16> %a, i16 %b, i32 %c) nounwind readnone {
;CHECK: variable_insertelement:
    %r = insertelement <8 x i16> %a, i16 %b, i32 %c
    ret <8 x i16> %r
}

declare void @llvm.arm.neon.vst4lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32, i32) nounwind

declare void @llvm.arm.neon.vst4lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32, i32) nounwind
