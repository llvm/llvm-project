; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define void @vst1i8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1i8:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vst1.8 {d16}, [r0:64]
	%tmp1 = load <8 x i8>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v8i8(ptr %A, <8 x i8> %tmp1, i32 16)
	ret void
}

define void @vst1i16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1i16:
;CHECK: vst1.16
	%tmp1 = load <4 x i16>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v4i16(ptr %A, <4 x i16> %tmp1, i32 1)
	ret void
}

define void @vst1i32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1i32:
;CHECK: vst1.32
	%tmp1 = load <2 x i32>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v2i32(ptr %A, <2 x i32> %tmp1, i32 1)
	ret void
}

define void @vst1f(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1f:
;CHECK: vst1.32
	%tmp1 = load <2 x float>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v2f32(ptr %A, <2 x float> %tmp1, i32 1)
	ret void
}

;Check for a post-increment updating store.
define void @vst1f_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst1f_update:
;CHECK: vst1.32 {d16}, [r{{[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%tmp1 = load <2 x float>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v2f32(ptr %A, <2 x float> %tmp1, i32 1)
	%tmp2 = getelementptr float, ptr %A, i32 2
	store ptr %tmp2, ptr %ptr
	ret void
}

define void @vst1i64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1i64:
;CHECK: vst1.64
	%tmp1 = load <1 x i64>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v1i64(ptr %A, <1 x i64> %tmp1, i32 1)
	ret void
}

define void @vst1Qi8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1Qi8:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vst1.8 {d16, d17}, [r0:64]
	%tmp1 = load <16 x i8>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v16i8(ptr %A, <16 x i8> %tmp1, i32 8)
	ret void
}

define void @vst1Qi16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1Qi16:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vst1.16 {d16, d17}, [r0:128]
	%tmp1 = load <8 x i16>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v8i16(ptr %A, <8 x i16> %tmp1, i32 32)
	ret void
}

;Check for a post-increment updating store with register increment.
define void @vst1Qi16_update(ptr %ptr, ptr %B, i32 %inc) nounwind {
;CHECK-LABEL: vst1Qi16_update:
;CHECK: vst1.16 {d16, d17}, [r1:64], r2
	%A = load ptr, ptr %ptr
	%tmp1 = load <8 x i16>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v8i16(ptr %A, <8 x i16> %tmp1, i32 8)
	%tmp2 = getelementptr i16, ptr %A, i32 %inc
	store ptr %tmp2, ptr %ptr
	ret void
}

define void @vst1Qi32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1Qi32:
;CHECK: vst1.32
	%tmp1 = load <4 x i32>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v4i32(ptr %A, <4 x i32> %tmp1, i32 1)
	ret void
}

define void @vst1Qf(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1Qf:
;CHECK: vst1.32
	%tmp1 = load <4 x float>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v4f32(ptr %A, <4 x float> %tmp1, i32 1)
	ret void
}

define void @vst1Qi64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1Qi64:
;CHECK: vst1.64
	%tmp1 = load <2 x i64>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v2i64(ptr %A, <2 x i64> %tmp1, i32 1)
	ret void
}

define void @vst1Qf64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst1Qf64:
;CHECK: vst1.64
	%tmp1 = load <2 x double>, ptr %B
	call void @llvm.arm.neon.vst1.p0.v2f64(ptr %A, <2 x double> %tmp1, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst1.p0.v8i8(ptr, <8 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v4i16(ptr, <4 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v2i32(ptr, <2 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v2f32(ptr, <2 x float>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v1i64(ptr, <1 x i64>, i32) nounwind

declare void @llvm.arm.neon.vst1.p0.v16i8(ptr, <16 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v8i16(ptr, <8 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v4i32(ptr, <4 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v4f32(ptr, <4 x float>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v2i64(ptr, <2 x i64>, i32) nounwind
declare void @llvm.arm.neon.vst1.p0.v2f64(ptr, <2 x double>, i32) nounwind
