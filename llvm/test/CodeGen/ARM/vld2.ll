; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

%struct.__neon_int8x8x2_t = type { <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x2_t = type { <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x2_t = type { <2 x float>, <2 x float> }
%struct.__neon_int64x1x2_t = type { <1 x i64>, <1 x i64> }

%struct.__neon_int8x16x2_t = type { <16 x i8>,  <16 x i8> }
%struct.__neon_int16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x2_t = type { <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x2_t = type { <4 x float>, <4 x float> }

define <8 x i8> @vld2i8(i8* %A) nounwind {
;CHECK-LABEL: vld2i8:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vld2.8 {d16, d17}, [{{r[0-9]+|lr}}:64]
	%tmp1 = call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2.v8i8.p0i8(i8* %A, i32 8)
        %tmp2 = extractvalue %struct.__neon_int8x8x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp1, 1
        %tmp4 = add <8 x i8> %tmp2, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vld2i16(i16* %A) nounwind {
;CHECK-LABEL: vld2i16:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vld2.16 {d16, d17}, [{{r[0-9]+|lr}}:128]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2.v4i16.p0i8(i8* %tmp0, i32 32)
        %tmp2 = extractvalue %struct.__neon_int16x4x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x4x2_t %tmp1, 1
        %tmp4 = add <4 x i16> %tmp2, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vld2i32(i32* %A) nounwind {
;CHECK-LABEL: vld2i32:
;CHECK: vld2.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2.v2i32.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_int32x2x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp1, 1
        %tmp4 = add <2 x i32> %tmp2, %tmp3
	ret <2 x i32> %tmp4
}

define <2 x float> @vld2f(float* %A) nounwind {
;CHECK-LABEL: vld2f:
;CHECK: vld2.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2.v2f32.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_float32x2x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x2x2_t %tmp1, 1
        %tmp4 = fadd <2 x float> %tmp2, %tmp3
	ret <2 x float> %tmp4
}

;Check for a post-increment updating load. 
define <2 x float> @vld2f_update(float** %ptr) nounwind {
;CHECK-LABEL: vld2f_update:
;CHECK: vld2.32 {d16, d17}, [{{r[0-9]+|lr}}]!
	%A = load float*, float** %ptr
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2.v2f32.p0i8(i8* %tmp0, i32 1)
	%tmp2 = extractvalue %struct.__neon_float32x2x2_t %tmp1, 0
	%tmp3 = extractvalue %struct.__neon_float32x2x2_t %tmp1, 1
	%tmp4 = fadd <2 x float> %tmp2, %tmp3
	%tmp5 = getelementptr float, float* %A, i32 4
	store float* %tmp5, float** %ptr
	ret <2 x float> %tmp4
}

define <1 x i64> @vld2i64(i64* %A) nounwind {
;CHECK-LABEL: vld2i64:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vld1.64 {d16, d17}, [{{r[0-9]+|lr}}:128]
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = call %struct.__neon_int64x1x2_t @llvm.arm.neon.vld2.v1i64.p0i8(i8* %tmp0, i32 32)
        %tmp2 = extractvalue %struct.__neon_int64x1x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int64x1x2_t %tmp1, 1
        %tmp4 = add <1 x i64> %tmp2, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @vld2Qi8(i8* %A) nounwind {
;CHECK-LABEL: vld2Qi8:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vld2.8 {d16, d17, d18, d19}, [{{r[0-9]+|lr}}:64]
	%tmp1 = call %struct.__neon_int8x16x2_t @llvm.arm.neon.vld2.v16i8.p0i8(i8* %A, i32 8)
        %tmp2 = extractvalue %struct.__neon_int8x16x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x16x2_t %tmp1, 1
        %tmp4 = add <16 x i8> %tmp2, %tmp3
	ret <16 x i8> %tmp4
}

;Check for a post-increment updating load with register increment.
define <16 x i8> @vld2Qi8_update(i8** %ptr, i32 %inc) nounwind {
;CHECK-LABEL: vld2Qi8_update:
;CHECK: vld2.8 {d16, d17, d18, d19}, [{{r[0-9]+|lr}}:128], r1
	%A = load i8*, i8** %ptr
	%tmp1 = call %struct.__neon_int8x16x2_t @llvm.arm.neon.vld2.v16i8.p0i8(i8* %A, i32 16)
        %tmp2 = extractvalue %struct.__neon_int8x16x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x16x2_t %tmp1, 1
        %tmp4 = add <16 x i8> %tmp2, %tmp3
	%tmp5 = getelementptr i8, i8* %A, i32 %inc
	store i8* %tmp5, i8** %ptr
	ret <16 x i8> %tmp4
}

define <8 x i16> @vld2Qi16(i16* %A) nounwind {
;CHECK-LABEL: vld2Qi16:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vld2.16 {d16, d17, d18, d19}, [{{r[0-9]+|lr}}:128]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2.v8i16.p0i8(i8* %tmp0, i32 16)
        %tmp2 = extractvalue %struct.__neon_int16x8x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x8x2_t %tmp1, 1
        %tmp4 = add <8 x i16> %tmp2, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vld2Qi32(i32* %A) nounwind {
;CHECK-LABEL: vld2Qi32:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vld2.32 {d16, d17, d18, d19}, [{{r[0-9]+|lr}}:256]
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32.p0i8(i8* %tmp0, i32 64)
        %tmp2 = extractvalue %struct.__neon_int32x4x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x4x2_t %tmp1, 1
        %tmp4 = add <4 x i32> %tmp2, %tmp3
	ret <4 x i32> %tmp4
}

define <4 x float> @vld2Qf(float* %A) nounwind {
;CHECK-LABEL: vld2Qf:
;CHECK: vld2.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call %struct.__neon_float32x4x2_t @llvm.arm.neon.vld2.v4f32.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_float32x4x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x4x2_t %tmp1, 1
        %tmp4 = fadd <4 x float> %tmp2, %tmp3
	ret <4 x float> %tmp4
}

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2.v8i8.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2.v4i16.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2.v2i32.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2.v2f32.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int64x1x2_t @llvm.arm.neon.vld2.v1i64.p0i8(i8*, i32) nounwind readonly

declare %struct.__neon_int8x16x2_t @llvm.arm.neon.vld2.v16i8.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2.v8i16.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_float32x4x2_t @llvm.arm.neon.vld2.v4f32.p0i8(i8*, i32) nounwind readonly
