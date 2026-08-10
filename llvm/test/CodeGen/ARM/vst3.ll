; RUN: llc -mtriple=arm-eabi -mattr=+neon -fast-isel=0 -O0 %s -o - | FileCheck %s

define void @vst3i8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3i8:
;Check the alignment value.  Max for this instruction is 64 bits:
;This test runs at -O0 so do not check for specific register numbers.
;CHECK: vst3.8 {d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}:64]
	%tmp1 = load <8 x i8>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v8i8(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 32)
	ret void
}

define void @vst3i16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3i16:
;CHECK: vst3.16
	%tmp1 = load <4 x i16>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v4i16(ptr %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1)
	ret void
}

define void @vst3i32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3i32:
;CHECK: vst3.32
	%tmp1 = load <2 x i32>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v2i32(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
	ret void
}

;Check for a post-increment updating store.
define void @vst3i32_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst3i32_update:
;CHECK: vst3.32 {d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}]!
	%A = load ptr, ptr %ptr
	%tmp1 = load <2 x i32>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v2i32(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
	%tmp2 = getelementptr i32, ptr %A, i32 6
	store ptr %tmp2, ptr %ptr
	ret void
}

define void @vst3f(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3f:
;CHECK: vst3.32
	%tmp1 = load <2 x float>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v2f32(ptr %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
	ret void
}

define void @vst3i64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3i64:
;Check the alignment value.  Max for this instruction is 64 bits:
;This test runs at -O0 so do not check for specific register numbers.
;CHECK: vst1.64 {d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}:64]
	%tmp1 = load <1 x i64>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v1i64(ptr %A, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 16)
	ret void
}

define void @vst3i64_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst3i64_update
;CHECK: vst1.64	{d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}]!
        %A = load ptr, ptr %ptr
        %tmp1 = load <1 x i64>, ptr %B
        call void @llvm.arm.neon.vst3.p0.v1i64(ptr %A, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 1)
        %tmp2 = getelementptr i64, ptr %A, i32 3
        store ptr %tmp2, ptr %ptr
        ret void
}

define void @vst3i64_reg_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst3i64_reg_update
;CHECK: vst1.64	{d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}], r{{.*}}
        %A = load ptr, ptr %ptr
        %tmp1 = load <1 x i64>, ptr %B
        call void @llvm.arm.neon.vst3.p0.v1i64(ptr %A, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 1)
        %tmp2 = getelementptr i64, ptr %A, i32 1
        store ptr %tmp2, ptr %ptr
        ret void
}

define void @vst3Qi8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3Qi8:
;Check the alignment value.  Max for this instruction is 64 bits:
;This test runs at -O0 so do not check for specific register numbers.
;CHECK: vst3.8 {d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}:64]!
;CHECK: vst3.8 {d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}:64]
	%tmp1 = load <16 x i8>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v16i8(ptr %A, <16 x i8> %tmp1, <16 x i8> %tmp1, <16 x i8> %tmp1, i32 32)
	ret void
}

define void @vst3Qi16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3Qi16:
;CHECK: vst3.16
;CHECK: vst3.16
	%tmp1 = load <8 x i16>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v8i16(ptr %A, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1)
	ret void
}

;Check for a post-increment updating store.
define void @vst3Qi16_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst3Qi16_update:
;CHECK: vst3.16 {d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}]!
;CHECK: vst3.16 {d{{.*}}, d{{.*}}, d{{.*}}}, [r{{.*}}]!
	%A = load ptr, ptr %ptr
	%tmp1 = load <8 x i16>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v8i16(ptr %A, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1)
	%tmp2 = getelementptr i16, ptr %A, i32 24
	store ptr %tmp2, ptr %ptr
	ret void
}

define void @vst3Qi32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3Qi32:
;CHECK: vst3.32
;CHECK: vst3.32
	%tmp1 = load <4 x i32>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v4i32(ptr %A, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 1)
	ret void
}

define void @vst3Qf(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst3Qf:
;CHECK: vst3.32
;CHECK: vst3.32
	%tmp1 = load <4 x float>, ptr %B
	call void @llvm.arm.neon.vst3.p0.v4f32(ptr %A, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst3.p0.v8i8(ptr, <8 x i8>, <8 x i8>, <8 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst3.p0.v4i16(ptr, <4 x i16>, <4 x i16>, <4 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst3.p0.v2i32(ptr, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst3.p0.v2f32(ptr, <2 x float>, <2 x float>, <2 x float>, i32) nounwind
declare void @llvm.arm.neon.vst3.p0.v1i64(ptr, <1 x i64>, <1 x i64>, <1 x i64>, i32) nounwind

declare void @llvm.arm.neon.vst3.p0.v16i8(ptr, <16 x i8>, <16 x i8>, <16 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst3.p0.v8i16(ptr, <8 x i16>, <8 x i16>, <8 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst3.p0.v4i32(ptr, <4 x i32>, <4 x i32>, <4 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst3.p0.v4f32(ptr, <4 x float>, <4 x float>, <4 x float>, i32) nounwind
