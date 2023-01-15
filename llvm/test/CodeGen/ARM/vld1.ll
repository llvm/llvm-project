; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+neon %s -o - | FileCheck %s

; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+neon -regalloc=basic %s -o - \
; RUN:	| FileCheck %s

define <8 x i8> @vld1i8(ptr %A) nounwind {
;CHECK-LABEL: vld1i8:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld1.8 {d16}, [r0:64]
	%tmp1 = call <8 x i8> @llvm.arm.neon.vld1.v8i8.p0(ptr %A, i32 16)
	ret <8 x i8> %tmp1
}

define <4 x i16> @vld1i16(ptr %A) nounwind {
;CHECK-LABEL: vld1i16:
;CHECK: vld1.16
	%tmp1 = call <4 x i16> @llvm.arm.neon.vld1.v4i16.p0(ptr %A, i32 1)
	ret <4 x i16> %tmp1
}

;Check for a post-increment updating load. 
define <4 x i16> @vld1i16_update(ptr %ptr) nounwind {
;CHECK-LABEL: vld1i16_update:
;CHECK: vld1.16 {d16}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%tmp1 = call <4 x i16> @llvm.arm.neon.vld1.v4i16.p0(ptr %A, i32 1)
	%tmp2 = getelementptr i16, ptr %A, i32 4
	       store ptr %tmp2, ptr %ptr
	ret <4 x i16> %tmp1
}

define <2 x i32> @vld1i32(ptr %A) nounwind {
;CHECK-LABEL: vld1i32:
;CHECK: vld1.32
	%tmp1 = call <2 x i32> @llvm.arm.neon.vld1.v2i32.p0(ptr %A, i32 1)
	ret <2 x i32> %tmp1
}

;Check for a post-increment updating load with register increment.
define <2 x i32> @vld1i32_update(ptr %ptr, i32 %inc) nounwind {
;CHECK-LABEL: vld1i32_update:
;CHECK: vld1.32 {d16}, [{{r[0-9]+}}], {{r[0-9]+}}
	%A = load ptr, ptr %ptr
	%tmp1 = call <2 x i32> @llvm.arm.neon.vld1.v2i32.p0(ptr %A, i32 1)
	%tmp2 = getelementptr i32, ptr %A, i32 %inc
	store ptr %tmp2, ptr %ptr
	ret <2 x i32> %tmp1
}

define <2 x float> @vld1f(ptr %A) nounwind {
;CHECK-LABEL: vld1f:
;CHECK: vld1.32
	%tmp1 = call <2 x float> @llvm.arm.neon.vld1.v2f32.p0(ptr %A, i32 1)
	ret <2 x float> %tmp1
}

define <1 x i64> @vld1i64(ptr %A) nounwind {
;CHECK-LABEL: vld1i64:
;CHECK: vld1.64
	%tmp1 = call <1 x i64> @llvm.arm.neon.vld1.v1i64.p0(ptr %A, i32 1)
	ret <1 x i64> %tmp1
}

define <16 x i8> @vld1Qi8(ptr %A) nounwind {
;CHECK-LABEL: vld1Qi8:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vld1.8 {d16, d17}, [r0:64]
	%tmp1 = call <16 x i8> @llvm.arm.neon.vld1.v16i8.p0(ptr %A, i32 8)
	ret <16 x i8> %tmp1
}

;Check for a post-increment updating load.
define <16 x i8> @vld1Qi8_update(ptr %ptr) nounwind {
;CHECK-LABEL: vld1Qi8_update:
;CHECK: vld1.8 {d16, d17}, [{{r[0-9]+|lr}}:64]!
	%A = load ptr, ptr %ptr
	%tmp1 = call <16 x i8> @llvm.arm.neon.vld1.v16i8.p0(ptr %A, i32 8)
	%tmp2 = getelementptr i8, ptr %A, i32 16
	store ptr %tmp2, ptr %ptr
	ret <16 x i8> %tmp1
}

define <8 x i16> @vld1Qi16(ptr %A) nounwind {
;CHECK-LABEL: vld1Qi16:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vld1.16 {d16, d17}, [r0:128]
	%tmp1 = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0(ptr %A, i32 32)
	ret <8 x i16> %tmp1
}

define <4 x i32> @vld1Qi32(ptr %A) nounwind {
;CHECK-LABEL: vld1Qi32:
;CHECK: vld1.32
	%tmp1 = call <4 x i32> @llvm.arm.neon.vld1.v4i32.p0(ptr %A, i32 1)
	ret <4 x i32> %tmp1
}

define <4 x float> @vld1Qf(ptr %A) nounwind {
;CHECK-LABEL: vld1Qf:
;CHECK: vld1.32
	%tmp1 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0(ptr %A, i32 1)
	ret <4 x float> %tmp1
}

define <2 x i64> @vld1Qi64(ptr %A) nounwind {
;CHECK-LABEL: vld1Qi64:
;CHECK: vld1.64
	%tmp1 = call <2 x i64> @llvm.arm.neon.vld1.v2i64.p0(ptr %A, i32 1)
	ret <2 x i64> %tmp1
}

define <2 x double> @vld1Qf64(ptr %A) nounwind {
;CHECK-LABEL: vld1Qf64:
;CHECK: vld1.64
	%tmp1 = call <2 x double> @llvm.arm.neon.vld1.v2f64.p0(ptr %A, i32 1)
	ret <2 x double> %tmp1
}

declare <8 x i8>  @llvm.arm.neon.vld1.v8i8.p0(ptr, i32) nounwind readonly
declare <4 x i16> @llvm.arm.neon.vld1.v4i16.p0(ptr, i32) nounwind readonly
declare <2 x i32> @llvm.arm.neon.vld1.v2i32.p0(ptr, i32) nounwind readonly
declare <2 x float> @llvm.arm.neon.vld1.v2f32.p0(ptr, i32) nounwind readonly
declare <1 x i64> @llvm.arm.neon.vld1.v1i64.p0(ptr, i32) nounwind readonly

declare <16 x i8> @llvm.arm.neon.vld1.v16i8.p0(ptr, i32) nounwind readonly
declare <8 x i16> @llvm.arm.neon.vld1.v8i16.p0(ptr, i32) nounwind readonly
declare <4 x i32> @llvm.arm.neon.vld1.v4i32.p0(ptr, i32) nounwind readonly
declare <4 x float> @llvm.arm.neon.vld1.v4f32.p0(ptr, i32) nounwind readonly
declare <2 x i64> @llvm.arm.neon.vld1.v2i64.p0(ptr, i32) nounwind readonly
declare <2 x double> @llvm.arm.neon.vld1.v2f64.p0(ptr, i32) nounwind readonly

; Radar 8355607
; Do not crash if the vld1 result is not used.
define void @unused_vld1_result() {
entry:
  %0 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0(ptr undef, i32 1)
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap() nounwind
