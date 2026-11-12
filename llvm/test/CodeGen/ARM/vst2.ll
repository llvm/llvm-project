; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define void @vst2i8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2i8:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vst2.8 {d16, d17}, [r0:64]
	%tmp1 = load <8 x i8>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v8i8(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 8)
	ret void
}

;Check for a post-increment updating store with register increment.
define void @vst2i8_update(ptr %ptr, ptr %B, i32 %inc) nounwind {
;CHECK-LABEL: vst2i8_update:
;CHECK: vst2.8 {d16, d17}, [r1], r2
	%A = load ptr, ptr %ptr
	%tmp1 = load <8 x i8>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v8i8(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 4)
	%tmp2 = getelementptr i8, ptr %A, i32 %inc
	store ptr %tmp2, ptr %ptr
	ret void
}

define void @vst2i16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2i16:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vst2.16 {d16, d17}, [r0:128]
	%tmp1 = load <4 x i16>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v4i16(ptr %A, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 32)
	ret void
}

define void @vst2i32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2i32:
;CHECK: vst2.32
	%tmp1 = load <2 x i32>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v2i32(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
	ret void
}

define void @vst2f(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2f:
;CHECK: vst2.32
	%tmp1 = load <2 x float>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v2f32(ptr %A, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
	ret void
}

define void @vst2i64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2i64:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vst1.64 {d16, d17}, [r0:128]
	%tmp1 = load <1 x i64>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v1i64(ptr %A, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 32)
	ret void
}

;Check for a post-increment updating store.
define void @vst2i64_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst2i64_update:
;CHECK: vst1.64 {d16, d17}, [r1:64]!
	%A = load ptr, ptr %ptr
	%tmp1 = load <1 x i64>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v1i64(ptr %A, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 8)
	%tmp2 = getelementptr i64, ptr %A, i32 2
	store ptr %tmp2, ptr %ptr
	ret void
}

define void @vst2Qi8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2Qi8:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst2.8 {d16, d17, d18, d19}, [r0:64]
	%tmp1 = load <16 x i8>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v16i8(ptr %A, <16 x i8> %tmp1, <16 x i8> %tmp1, i32 8)
	ret void
}

define void @vst2Qi16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2Qi16:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst2.16 {d16, d17, d18, d19}, [r0:128]
	%tmp1 = load <8 x i16>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v8i16(ptr %A, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 16)
	ret void
}

define void @vst2Qi32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2Qi32:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst2.32 {d16, d17, d18, d19}, [r0:256]
	%tmp1 = load <4 x i32>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v4i32(ptr %A, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 64)
	ret void
}

define void @vst2Qf(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst2Qf:
;CHECK: vst2.32
	%tmp1 = load <4 x float>, ptr %B
	call void @llvm.arm.neon.vst2.p0.v4f32(ptr %A, <4 x float> %tmp1, <4 x float> %tmp1, i32 1)
	ret void
}

define ptr @vst2update(ptr %out, ptr %B) nounwind {
;CHECK-LABEL: vst2update:
;CHECK: vst2.16 {d16, d17}, [r0]!
	%tmp1 = load <4 x i16>, ptr %B
	tail call void @llvm.arm.neon.vst2.p0.v4i16(ptr %out, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 2)
	%t5 = getelementptr inbounds i8, ptr %out, i32 16
	ret ptr %t5
}

define ptr @vst2update2(ptr %out, ptr %this) nounwind optsize ssp align 2 {
;CHECK-LABEL: vst2update2:
;CHECK: vst2.32 {d16, d17, d18, d19}, [r0]!
  %tmp1 = load <4 x float>, ptr %this
  call void @llvm.arm.neon.vst2.p0.v4f32(ptr %out, <4 x float> %tmp1, <4 x float> %tmp1, i32 4) nounwind
  %tmp2 = getelementptr inbounds i8, ptr %out, i32  32
  ret ptr %tmp2
}

declare void @llvm.arm.neon.vst2.p0.v8i8(ptr, <8 x i8>, <8 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst2.p0.v4i16(ptr, <4 x i16>, <4 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst2.p0.v2i32(ptr, <2 x i32>, <2 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst2.p0.v2f32(ptr, <2 x float>, <2 x float>, i32) nounwind
declare void @llvm.arm.neon.vst2.p0.v1i64(ptr, <1 x i64>, <1 x i64>, i32) nounwind

declare void @llvm.arm.neon.vst2.p0.v16i8(ptr, <16 x i8>, <16 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst2.p0.v8i16(ptr, <8 x i16>, <8 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst2.p0.v4i32(ptr, <4 x i32>, <4 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst2.p0.v4f32(ptr, <4 x float>, <4 x float>, i32) nounwind
