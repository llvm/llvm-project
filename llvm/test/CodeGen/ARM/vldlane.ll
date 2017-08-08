; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+neon %s -o - | FileCheck %s

; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+neon -regalloc=basic %s -o - \
; RUN:	| FileCheck %s

define <8 x i8> @vld1lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vld1lanei8:
;Check the (default) alignment value.
;CHECK: vld1.8 {d16[3]}, [r0]
	%tmp1 = load <8 x i8>, <8 x i8>* %B
	%tmp2 = load i8, i8* %A, align 8
	%tmp3 = insertelement <8 x i8> %tmp1, i8 %tmp2, i32 3
        ret <8 x i8> %tmp3
}

define <4 x i16> @vld1lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vld1lanei16:
;Check the alignment value.  Max for this instruction is 16 bits:
;CHECK: vld1.16 {d16[2]}, [r0:16]
	%tmp1 = load <4 x i16>, <4 x i16>* %B
	%tmp2 = load i16, i16* %A, align 8
	%tmp3 = insertelement <4 x i16> %tmp1, i16 %tmp2, i32 2
        ret <4 x i16> %tmp3
}

define <2 x i32> @vld1lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vld1lanei32:
;Check the alignment value.  Max for this instruction is 32 bits:
;CHECK: vld1.32 {d16[1]}, [r0:32]
	%tmp1 = load <2 x i32>, <2 x i32>* %B
	%tmp2 = load i32, i32* %A, align 8
	%tmp3 = insertelement <2 x i32> %tmp1, i32 %tmp2, i32 1
        ret <2 x i32> %tmp3
}

define <2 x i32> @vld1lanei32a32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vld1lanei32a32:
;Check the alignment value.  Legal values are none or :32.
;CHECK: vld1.32 {d16[1]}, [r0:32]
	%tmp1 = load <2 x i32>, <2 x i32>* %B
	%tmp2 = load i32, i32* %A, align 4
	%tmp3 = insertelement <2 x i32> %tmp1, i32 %tmp2, i32 1
        ret <2 x i32> %tmp3
}

define <2 x float> @vld1lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vld1lanef:
;CHECK: vld1.32 {d16[1]}, [r0:32]
	%tmp1 = load <2 x float>, <2 x float>* %B
	%tmp2 = load float, float* %A, align 4
	%tmp3 = insertelement <2 x float> %tmp1, float %tmp2, i32 1
	ret <2 x float> %tmp3
}

define <16 x i8> @vld1laneQi8(i8* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vld1laneQi8:
;CHECK: vld1.8 {d17[1]}, [r0]
	%tmp1 = load <16 x i8>, <16 x i8>* %B
	%tmp2 = load i8, i8* %A, align 8
	%tmp3 = insertelement <16 x i8> %tmp1, i8 %tmp2, i32 9
	ret <16 x i8> %tmp3
}

define <8 x i16> @vld1laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vld1laneQi16:
;CHECK: vld1.16 {d17[1]}, [r0:16]
	%tmp1 = load <8 x i16>, <8 x i16>* %B
	%tmp2 = load i16, i16* %A, align 8
	%tmp3 = insertelement <8 x i16> %tmp1, i16 %tmp2, i32 5
	ret <8 x i16> %tmp3
}

define <4 x i32> @vld1laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vld1laneQi32:
;CHECK: vld1.32 {d17[1]}, [r0:32]
	%tmp1 = load <4 x i32>, <4 x i32>* %B
	%tmp2 = load i32, i32* %A, align 8
	%tmp3 = insertelement <4 x i32> %tmp1, i32 %tmp2, i32 3
	ret <4 x i32> %tmp3
}

define <4 x float> @vld1laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vld1laneQf:
;CHECK: vld1.32 {d16[0]}, [r0:32]
	%tmp1 = load <4 x float>, <4 x float>* %B
	%tmp2 = load float, float* %A
	%tmp3 = insertelement <4 x float> %tmp1, float %tmp2, i32 0
	ret <4 x float> %tmp3
}

%struct.__neon_int8x8x2_t = type { <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x2_t = type { <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x2_t = type { <2 x float>, <2 x float> }

%struct.__neon_int16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x2_t = type { <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x2_t = type { <4 x float>, <4 x float> }

define <8 x i8> @vld2lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vld2lanei8:
;Check the alignment value.  Max for this instruction is 16 bits:
;CHECK: vld2.8 {d16[1], d17[1]}, [r0:16]
	%tmp1 = load <8 x i8>, <8 x i8>* %B
	%tmp2 = call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 4)
        %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vld2lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vld2lanei16:
;Check the alignment value.  Max for this instruction is 32 bits:
;CHECK: vld2.16 {d16[1], d17[1]}, [r0:32]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>, <4 x i16>* %B
	%tmp2 = call %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2lane.v4i16.p0i8(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 8)
        %tmp3 = extractvalue %struct.__neon_int16x4x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x4x2_t %tmp2, 1
        %tmp5 = add <4 x i16> %tmp3, %tmp4
	ret <4 x i16> %tmp5
}

define <2 x i32> @vld2lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vld2lanei32:
;CHECK: vld2.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>, <2 x i32>* %B
	%tmp2 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32.p0i8(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 1
        %tmp5 = add <2 x i32> %tmp3, %tmp4
	ret <2 x i32> %tmp5
}

;Check for a post-increment updating load.
define <2 x i32> @vld2lanei32_update(i32** %ptr, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vld2lanei32_update:
;CHECK: vld2.32 {d16[1], d17[1]}, [{{r[0-9]+}}]!
	%A = load i32*, i32** %ptr
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>, <2 x i32>* %B
	%tmp2 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32.p0i8(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
	%tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 0
	%tmp4 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 1
	%tmp5 = add <2 x i32> %tmp3, %tmp4
	%tmp6 = getelementptr i32, i32* %A, i32 2
	store i32* %tmp6, i32** %ptr
	ret <2 x i32> %tmp5
}

define <2 x i32> @vld2lanei32_odd_update(i32** %ptr, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vld2lanei32_odd_update:
;CHECK: mov [[INC:r[0-9]+]], #12
;CHECK: vld2.32 {d16[1], d17[1]}, [{{r[0-9]+}}], [[INC]]
	%A = load i32*, i32** %ptr
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>, <2 x i32>* %B
	%tmp2 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32.p0i8(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
	%tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 0
	%tmp4 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 1
	%tmp5 = add <2 x i32> %tmp3, %tmp4
	%tmp6 = getelementptr i32, i32* %A, i32 3
	store i32* %tmp6, i32** %ptr
	ret <2 x i32> %tmp5
}

define <2 x float> @vld2lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vld2lanef:
;CHECK: vld2.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>, <2 x float>* %B
	%tmp2 = call %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2lane.v2f32.p0i8(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x2x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x2x2_t %tmp2, 1
        %tmp5 = fadd <2 x float> %tmp3, %tmp4
	ret <2 x float> %tmp5
}

define <8 x i16> @vld2laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vld2laneQi16:
;Check the (default) alignment.
;CHECK: vld2.16 {d17[1], d19[1]}, [{{r[0-9]+}}]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>, <8 x i16>* %B
	%tmp2 = call %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16.p0i8(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 5, i32 1)
        %tmp3 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 1
        %tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vld2laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vld2laneQi32:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld2.32 {d17[0], d19[0]}, [{{r[0-9]+}}:64]
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>, <4 x i32>* %B
	%tmp2 = call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2lane.v4i32.p0i8(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2, i32 16)
        %tmp3 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 1
        %tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x float> @vld2laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vld2laneQf:
;CHECK: vld2.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>, <4 x float>* %B
	%tmp2 = call %struct.__neon_float32x4x2_t @llvm.arm.neon.vld2lane.v4f32.p0i8(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x4x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x4x2_t %tmp2, 1
        %tmp5 = fadd <4 x float> %tmp3, %tmp4
	ret <4 x float> %tmp5
}

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8*, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2lane.v4i16.p0i8(i8*, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32.p0i8(i8*, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2lane.v2f32.p0i8(i8*, <2 x float>, <2 x float>, i32, i32) nounwind readonly

declare %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16.p0i8(i8*, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2lane.v4i32.p0i8(i8*, <4 x i32>, <4 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x4x2_t @llvm.arm.neon.vld2lane.v4f32.p0i8(i8*, <4 x float>, <4 x float>, i32, i32) nounwind readonly

%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x3_t = type { <2 x float>, <2 x float>, <2 x float> }

%struct.__neon_int16x8x3_t = type { <8 x i16>, <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x3_t = type { <4 x i32>, <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x3_t = type { <4 x float>, <4 x float>, <4 x float> }

define <8 x i8> @vld3lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vld3lanei8:
;CHECK: vld3.8
	%tmp1 = load <8 x i8>, <8 x i8>* %B
	%tmp2 = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3lane.v8i8.p0i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 2
        %tmp6 = add <8 x i8> %tmp3, %tmp4
        %tmp7 = add <8 x i8> %tmp5, %tmp6
	ret <8 x i8> %tmp7
}

define <4 x i16> @vld3lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vld3lanei16:
;Check the (default) alignment value.  VLD3 does not support alignment.
;CHECK: vld3.16 {d{{.*}}[1], d{{.*}}[1], d{{.*}}[1]}, [{{r[0-9]+}}]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>, <4 x i16>* %B
	%tmp2 = call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16.p0i8(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 8)
        %tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 2
        %tmp6 = add <4 x i16> %tmp3, %tmp4
        %tmp7 = add <4 x i16> %tmp5, %tmp6
	ret <4 x i16> %tmp7
}

define <2 x i32> @vld3lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vld3lanei32:
;CHECK: vld3.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>, <2 x i32>* %B
	%tmp2 = call %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3lane.v2i32.p0i8(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 2
        %tmp6 = add <2 x i32> %tmp3, %tmp4
        %tmp7 = add <2 x i32> %tmp5, %tmp6
	ret <2 x i32> %tmp7
}

define <2 x float> @vld3lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vld3lanef:
;CHECK: vld3.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>, <2 x float>* %B
	%tmp2 = call %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3lane.v2f32.p0i8(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 2
        %tmp6 = fadd <2 x float> %tmp3, %tmp4
        %tmp7 = fadd <2 x float> %tmp5, %tmp6
	ret <2 x float> %tmp7
}

define <8 x i16> @vld3laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vld3laneQi16:
;Check the (default) alignment value.  VLD3 does not support alignment.
;CHECK: vld3.16 {d{{.*}}[1], d{{.*}}[1], d{{.*}}[1]}, [{{r[0-9]+}}]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>, <8 x i16>* %B
	%tmp2 = call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3lane.v8i16.p0i8(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1, i32 8)
        %tmp3 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 2
        %tmp6 = add <8 x i16> %tmp3, %tmp4
        %tmp7 = add <8 x i16> %tmp5, %tmp6
	ret <8 x i16> %tmp7
}

;Check for a post-increment updating load with register increment.
define <8 x i16> @vld3laneQi16_update(i16** %ptr, <8 x i16>* %B, i32 %inc) nounwind {
;CHECK-LABEL: vld3laneQi16_update:
;CHECK: vld3.16 {d{{.*}}[1], d{{.*}}[1], d{{.*}}[1]}, [{{r[0-9]+|lr}}], {{r[0-9]+}}
	%A = load i16*, i16** %ptr
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>, <8 x i16>* %B
	%tmp2 = call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3lane.v8i16.p0i8(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1, i32 8)
	%tmp3 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 0
	%tmp4 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 1
	%tmp5 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 2
	%tmp6 = add <8 x i16> %tmp3, %tmp4
	%tmp7 = add <8 x i16> %tmp5, %tmp6
	%tmp8 = getelementptr i16, i16* %A, i32 %inc
	store i16* %tmp8, i16** %ptr
	ret <8 x i16> %tmp7
}

define <4 x i32> @vld3laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vld3laneQi32:
;CHECK: vld3.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>, <4 x i32>* %B
	%tmp2 = call %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3lane.v4i32.p0i8(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 3, i32 1)
        %tmp3 = extractvalue %struct.__neon_int32x4x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x4x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int32x4x3_t %tmp2, 2
        %tmp6 = add <4 x i32> %tmp3, %tmp4
        %tmp7 = add <4 x i32> %tmp5, %tmp6
	ret <4 x i32> %tmp7
}

define <4 x float> @vld3laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vld3laneQf:
;CHECK: vld3.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>, <4 x float>* %B
	%tmp2 = call %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3lane.v4f32.p0i8(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x4x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x4x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_float32x4x3_t %tmp2, 2
        %tmp6 = fadd <4 x float> %tmp3, %tmp4
        %tmp7 = fadd <4 x float> %tmp5, %tmp6
	ret <4 x float> %tmp7
}

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3lane.v8i8.p0i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16.p0i8(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3lane.v2i32.p0i8(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3lane.v2f32.p0i8(i8*, <2 x float>, <2 x float>, <2 x float>, i32, i32) nounwind readonly

declare %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3lane.v8i16.p0i8(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3lane.v4i32.p0i8(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3lane.v4f32.p0i8(i8*, <4 x float>, <4 x float>, <4 x float>, i32, i32) nounwind readonly

%struct.__neon_int8x8x4_t = type { <8 x i8>,  <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x4_t = type { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x4_t = type { <2 x float>, <2 x float>, <2 x float>, <2 x float> }

%struct.__neon_int16x8x4_t = type { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x4_t = type { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x4_t = type { <4 x float>, <4 x float>, <4 x float>, <4 x float> }

define <8 x i8> @vld4lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vld4lanei8:
;Check the alignment value.  Max for this instruction is 32 bits:
;CHECK: vld4.8 {d{{.*}}[1], d{{.*}}[1], d{{.*}}[1], d{{.*}}[1]}, [{{r[0-9]+}}:32]
	%tmp1 = load <8 x i8>, <8 x i8>* %B
	%tmp2 = call %struct.__neon_int8x8x4_t @llvm.arm.neon.vld4lane.v8i8.p0i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 8)
        %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
        %tmp7 = add <8 x i8> %tmp3, %tmp4
        %tmp8 = add <8 x i8> %tmp5, %tmp6
        %tmp9 = add <8 x i8> %tmp7, %tmp8
	ret <8 x i8> %tmp9
}

;Check for a post-increment updating load.
define <8 x i8> @vld4lanei8_update(i8** %ptr, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vld4lanei8_update:
;CHECK: vld4.8 {d16[1], d17[1], d18[1], d19[1]}, [{{r[0-9]+}}:32]!
	%A = load i8*, i8** %ptr
	%tmp1 = load <8 x i8>, <8 x i8>* %B
	%tmp2 = call %struct.__neon_int8x8x4_t @llvm.arm.neon.vld4lane.v8i8.p0i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 8)
	%tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
	%tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
	%tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
	%tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
	%tmp7 = add <8 x i8> %tmp3, %tmp4
	%tmp8 = add <8 x i8> %tmp5, %tmp6
	%tmp9 = add <8 x i8> %tmp7, %tmp8
	%tmp10 = getelementptr i8, i8* %A, i32 4
	store i8* %tmp10, i8** %ptr
	ret <8 x i8> %tmp9
}

define <4 x i16> @vld4lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vld4lanei16:
;Check that a power-of-two alignment smaller than the total size of the memory
;being loaded is ignored.
;CHECK: vld4.16 {d16[1], d17[1], d18[1], d19[1]}, [{{r[0-9]+}}]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>, <4 x i16>* %B
	%tmp2 = call %struct.__neon_int16x4x4_t @llvm.arm.neon.vld4lane.v4i16.p0i8(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 4)
        %tmp3 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 3
        %tmp7 = add <4 x i16> %tmp3, %tmp4
        %tmp8 = add <4 x i16> %tmp5, %tmp6
        %tmp9 = add <4 x i16> %tmp7, %tmp8
	ret <4 x i16> %tmp9
}

define <2 x i32> @vld4lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vld4lanei32:
;Check the alignment value.  An 8-byte alignment is allowed here even though
;it is smaller than the total size of the memory being loaded.
;CHECK: vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [{{r[0-9]+}}:64]
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>, <2 x i32>* %B
	%tmp2 = call %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32.p0i8(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 8)
        %tmp3 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 3
        %tmp7 = add <2 x i32> %tmp3, %tmp4
        %tmp8 = add <2 x i32> %tmp5, %tmp6
        %tmp9 = add <2 x i32> %tmp7, %tmp8
	ret <2 x i32> %tmp9
}

define <2 x float> @vld4lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vld4lanef:
;CHECK: vld4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>, <2 x float>* %B
	%tmp2 = call %struct.__neon_float32x2x4_t @llvm.arm.neon.vld4lane.v2f32.p0i8(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 3
        %tmp7 = fadd <2 x float> %tmp3, %tmp4
        %tmp8 = fadd <2 x float> %tmp5, %tmp6
        %tmp9 = fadd <2 x float> %tmp7, %tmp8
	ret <2 x float> %tmp9
}

define <8 x i16> @vld4laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vld4laneQi16:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld4.16 {d16[1], d18[1], d20[1], d22[1]}, [{{r[0-9]+}}:64]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>, <8 x i16>* %B
	%tmp2 = call %struct.__neon_int16x8x4_t @llvm.arm.neon.vld4lane.v8i16.p0i8(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1, i32 16)
        %tmp3 = extractvalue %struct.__neon_int16x8x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x8x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int16x8x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int16x8x4_t %tmp2, 3
        %tmp7 = add <8 x i16> %tmp3, %tmp4
        %tmp8 = add <8 x i16> %tmp5, %tmp6
        %tmp9 = add <8 x i16> %tmp7, %tmp8
	ret <8 x i16> %tmp9
}

define <4 x i32> @vld4laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vld4laneQi32:
;Check the (default) alignment.
;CHECK: vld4.32 {d17[0], d19[0], d21[0], d23[0]}, [{{r[0-9]+}}]
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>, <4 x i32>* %B
	%tmp2 = call %struct.__neon_int32x4x4_t @llvm.arm.neon.vld4lane.v4i32.p0i8(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2, i32 1)
        %tmp3 = extractvalue %struct.__neon_int32x4x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x4x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int32x4x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int32x4x4_t %tmp2, 3
        %tmp7 = add <4 x i32> %tmp3, %tmp4
        %tmp8 = add <4 x i32> %tmp5, %tmp6
        %tmp9 = add <4 x i32> %tmp7, %tmp8
	ret <4 x i32> %tmp9
}

define <4 x float> @vld4laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vld4laneQf:
;CHECK: vld4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>, <4 x float>* %B
	%tmp2 = call %struct.__neon_float32x4x4_t @llvm.arm.neon.vld4lane.v4f32.p0i8(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x4x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x4x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_float32x4x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_float32x4x4_t %tmp2, 3
        %tmp7 = fadd <4 x float> %tmp3, %tmp4
        %tmp8 = fadd <4 x float> %tmp5, %tmp6
        %tmp9 = fadd <4 x float> %tmp7, %tmp8
	ret <4 x float> %tmp9
}

declare %struct.__neon_int8x8x4_t @llvm.arm.neon.vld4lane.v8i8.p0i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int16x4x4_t @llvm.arm.neon.vld4lane.v4i16.p0i8(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32.p0i8(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x2x4_t @llvm.arm.neon.vld4lane.v2f32.p0i8(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32, i32) nounwind readonly

declare %struct.__neon_int16x8x4_t @llvm.arm.neon.vld4lane.v8i16.p0i8(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x4x4_t @llvm.arm.neon.vld4lane.v4i32.p0i8(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x4x4_t @llvm.arm.neon.vld4lane.v4f32.p0i8(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32, i32) nounwind readonly

; Radar 8776599: If one of the operands to a QQQQ REG_SEQUENCE is a register
; in the QPR_VFP2 regclass, it needs to be copied to a QPR regclass because
; we don't currently have a QQQQ_VFP2 super-regclass.  (The "0" for the low
; part of %ins67 is supposed to be loaded by a VLDRS instruction in this test.)
define <8 x i16> @test_qqqq_regsequence_subreg([6 x i64] %b) nounwind {
;CHECK-LABEL: test_qqqq_regsequence_subreg:
;CHECK: vld3.16
  %tmp63 = extractvalue [6 x i64] %b, 5
  %tmp64 = zext i64 %tmp63 to i128
  %tmp65 = shl i128 %tmp64, 64
  %ins67 = or i128 %tmp65, 0
  %tmp78 = bitcast i128 %ins67 to <8 x i16>
  %vld3_lane = tail call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3lane.v8i16.p0i8(i8* undef, <8 x i16> undef, <8 x i16> undef, <8 x i16> %tmp78, i32 1, i32 2)
  %tmp3 = extractvalue %struct.__neon_int16x8x3_t %vld3_lane, 0
  %tmp4 = extractvalue %struct.__neon_int16x8x3_t %vld3_lane, 1
  %tmp5 = extractvalue %struct.__neon_int16x8x3_t %vld3_lane, 2
  %tmp6 = add <8 x i16> %tmp3, %tmp4
  %tmp7 = add <8 x i16> %tmp5, %tmp6
  ret <8 x i16> %tmp7
}

declare void @llvm.trap() nounwind
