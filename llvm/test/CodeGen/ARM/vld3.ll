; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o -| FileCheck %s
; RUN: llc -mtriple=arm-eabi -mattr=+neon -regalloc=basic %s -o - | FileCheck %s

%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x3_t = type { <2 x float>, <2 x float>, <2 x float> }
%struct.__neon_int64x1x3_t = type { <1 x i64>, <1 x i64>, <1 x i64> }

%struct.__neon_int8x16x3_t = type { <16 x i8>,  <16 x i8>,  <16 x i8> }
%struct.__neon_int16x8x3_t = type { <8 x i16>, <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x3_t = type { <4 x i32>, <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x3_t = type { <4 x float>, <4 x float>, <4 x float> }

define <8 x i8> @vld3i8(ptr %A) nounwind {
;CHECK-LABEL: vld3i8:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld3.8 {d16, d17, d18}, [{{r[0-9]+|lr}}:64]
	%tmp1 = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr %A, i32 32)
        %tmp2 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 2
        %tmp4 = add <8 x i8> %tmp2, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vld3i16(ptr %A) nounwind {
;CHECK-LABEL: vld3i16:
;CHECK: vld3.16
	%tmp1 = call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3.v4i16.p0(ptr %A, i32 1)
        %tmp2 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 2
        %tmp4 = add <4 x i16> %tmp2, %tmp3
	ret <4 x i16> %tmp4
}

;Check for a post-increment updating load with register increment.
define <4 x i16> @vld3i16_update(ptr %ptr, i32 %inc) nounwind {
;CHECK-LABEL: vld3i16_update:
;CHECK: vld3.16 {d16, d17, d18}, [{{r[0-9]+|lr}}], {{r[0-9]+|lr}}
	%A = load ptr, ptr %ptr
	%tmp1 = call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3.v4i16.p0(ptr %A, i32 1)
	%tmp2 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 0
	%tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 2
	%tmp4 = add <4 x i16> %tmp2, %tmp3
	%tmp5 = getelementptr i16, ptr %A, i32 %inc
	store ptr %tmp5, ptr %ptr
	ret <4 x i16> %tmp4
}

define <2 x i32> @vld3i32(ptr %A) nounwind {
;CHECK-LABEL: vld3i32:
;CHECK: vld3.32
	%tmp1 = call %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3.v2i32.p0(ptr %A, i32 1)
        %tmp2 = extractvalue %struct.__neon_int32x2x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x2x3_t %tmp1, 2
        %tmp4 = add <2 x i32> %tmp2, %tmp3
	ret <2 x i32> %tmp4
}

define <2 x float> @vld3f(ptr %A) nounwind {
;CHECK-LABEL: vld3f:
;CHECK: vld3.32
	%tmp1 = call %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3.v2f32.p0(ptr %A, i32 1)
        %tmp2 = extractvalue %struct.__neon_float32x2x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x2x3_t %tmp1, 2
        %tmp4 = fadd <2 x float> %tmp2, %tmp3
	ret <2 x float> %tmp4
}

define <1 x i64> @vld3i64(ptr %A) nounwind {
;CHECK-LABEL: vld3i64:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld1.64 {d16, d17, d18}, [{{r[0-9]+|lr}}:64]
	%tmp1 = call %struct.__neon_int64x1x3_t @llvm.arm.neon.vld3.v1i64.p0(ptr %A, i32 16)
        %tmp2 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 2
        %tmp4 = add <1 x i64> %tmp2, %tmp3
	ret <1 x i64> %tmp4
}

define <1 x i64> @vld3i64_update(ptr %ptr, ptr %A) nounwind {
;CHECK-LABEL: vld3i64_update:
;CHECK: vld1.64	{d16, d17, d18}, [{{r[0-9]+|lr}}:64]!
        %tmp1 = call %struct.__neon_int64x1x3_t @llvm.arm.neon.vld3.v1i64.p0(ptr %A, i32 16)
        %tmp5 = getelementptr i64, ptr %A, i32 3
        store ptr %tmp5, ptr %ptr
        %tmp2 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 2
        %tmp4 = add <1 x i64> %tmp2, %tmp3
        ret <1 x i64> %tmp4
}

define <1 x i64> @vld3i64_reg_update(ptr %ptr, ptr %A) nounwind {
;CHECK-LABEL: vld3i64_reg_update:
;CHECK: vld1.64	{d16, d17, d18}, [{{r[0-9]+|lr}}:64], {{r[0-9]+|lr}}
        %tmp1 = call %struct.__neon_int64x1x3_t @llvm.arm.neon.vld3.v1i64.p0(ptr %A, i32 16)
        %tmp5 = getelementptr i64, ptr %A, i32 1
        store ptr %tmp5, ptr %ptr
        %tmp2 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 2
        %tmp4 = add <1 x i64> %tmp2, %tmp3
        ret <1 x i64> %tmp4
}

define <16 x i8> @vld3Qi8(ptr %A) nounwind {
;CHECK-LABEL: vld3Qi8:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld3.8 {d16, d18, d20}, [{{r[0-9]+|lr}}:64]!
;CHECK: vld3.8 {d17, d19, d21}, [{{r[0-9]+|lr}}:64]
	%tmp1 = call %struct.__neon_int8x16x3_t @llvm.arm.neon.vld3.v16i8.p0(ptr %A, i32 32)
        %tmp2 = extractvalue %struct.__neon_int8x16x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x16x3_t %tmp1, 2
        %tmp4 = add <16 x i8> %tmp2, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vld3Qi16(ptr %A) nounwind {
;CHECK-LABEL: vld3Qi16:
;CHECK: vld3.16
;CHECK: vld3.16
	%tmp1 = call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3.v8i16.p0(ptr %A, i32 1)
        %tmp2 = extractvalue %struct.__neon_int16x8x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x8x3_t %tmp1, 2
        %tmp4 = add <8 x i16> %tmp2, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vld3Qi32(ptr %A) nounwind {
;CHECK-LABEL: vld3Qi32:
;CHECK: vld3.32
;CHECK: vld3.32
	%tmp1 = call %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3.v4i32.p0(ptr %A, i32 1)
        %tmp2 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 2
        %tmp4 = add <4 x i32> %tmp2, %tmp3
	ret <4 x i32> %tmp4
}

;Check for a post-increment updating load. 
define <4 x i32> @vld3Qi32_update(ptr %ptr) nounwind {
;CHECK-LABEL: vld3Qi32_update:
;CHECK: vld3.32 {d16, d18, d20}, [[[R:r[0-9]+|lr]]]!
;CHECK: vld3.32 {d17, d19, d21}, [[[R]]]!
	%A = load ptr, ptr %ptr
	%tmp1 = call %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3.v4i32.p0(ptr %A, i32 1)
	%tmp2 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 0
	%tmp3 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 2
	%tmp4 = add <4 x i32> %tmp2, %tmp3
	%tmp5 = getelementptr i32, ptr %A, i32 12
	store ptr %tmp5, ptr %ptr
	ret <4 x i32> %tmp4
}

define <4 x float> @vld3Qf(ptr %A) nounwind {
;CHECK-LABEL: vld3Qf:
;CHECK: vld3.32
;CHECK: vld3.32
	%tmp1 = call %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3.v4f32.p0(ptr %A, i32 1)
        %tmp2 = extractvalue %struct.__neon_float32x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x4x3_t %tmp1, 2
        %tmp4 = fadd <4 x float> %tmp2, %tmp3
	ret <4 x float> %tmp4
}

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr, i32) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3.v4i16.p0(ptr, i32) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3.v2i32.p0(ptr, i32) nounwind readonly
declare %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3.v2f32.p0(ptr, i32) nounwind readonly
declare %struct.__neon_int64x1x3_t @llvm.arm.neon.vld3.v1i64.p0(ptr, i32) nounwind readonly

declare %struct.__neon_int8x16x3_t @llvm.arm.neon.vld3.v16i8.p0(ptr, i32) nounwind readonly
declare %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3.v8i16.p0(ptr, i32) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3.v4i32.p0(ptr, i32) nounwind readonly
declare %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3.v4f32.p0(ptr, i32) nounwind readonly
