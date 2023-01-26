; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define void @vst4i8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4i8:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst4.8 {d16, d17, d18, d19}, [r0:64]
	%tmp1 = load <8 x i8>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v8i8(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 8)
	ret void
}

;Check for a post-increment updating store with register increment.
define void @vst4i8_update(ptr %ptr, ptr %B, i32 %inc) nounwind {
;CHECK-LABEL: vst4i8_update:
;CHECK: vst4.8 {d16, d17, d18, d19}, [r{{[0-9]+}}:128], r2
	%A = load ptr, ptr %ptr
	%tmp1 = load <8 x i8>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v8i8(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 16)
	%tmp2 = getelementptr i8, ptr %A, i32 %inc
	store ptr %tmp2, ptr %ptr
	ret void
}

define void @vst4i16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4i16:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst4.16 {d16, d17, d18, d19}, [r0:128]
	%tmp1 = load <4 x i16>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v4i16(ptr %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 16)
	ret void
}

define void @vst4i32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4i32:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst4.32 {d16, d17, d18, d19}, [r0:256]
	%tmp1 = load <2 x i32>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v2i32(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 32)
	ret void
}

define void @vst4f(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4f:
;CHECK: vst4.32
	%tmp1 = load <2 x float>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v2f32(ptr %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
	ret void
}

define void @vst4i64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4i64:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst1.64 {d16, d17, d18, d19}, [r0:256]
	%tmp1 = load <1 x i64>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v1i64(ptr %A, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 64)
	ret void
}

define void @vst4i64_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst4i64_update:
;CHECK: vst1.64	{d16, d17, d18, d19}, [r{{[0-9]+}}]!
        %A = load ptr, ptr %ptr
        %tmp1 = load <1 x i64>, ptr %B
        call void @llvm.arm.neon.vst4.p0.v1i64(ptr %A, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 1)
        %tmp2 = getelementptr i64, ptr %A, i32 4
        store ptr %tmp2, ptr %ptr
        ret void
}

define void @vst4i64_reg_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst4i64_reg_update:
;CHECK: vst1.64	{d16, d17, d18, d19}, [r{{[0-9]+}}], r{{[0-9]+}}
        %A = load ptr, ptr %ptr
        %tmp1 = load <1 x i64>, ptr %B
        call void @llvm.arm.neon.vst4.p0.v1i64(ptr %A, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 1)
        %tmp2 = getelementptr i64, ptr %A, i32 1
        store ptr %tmp2, ptr %ptr
        ret void
}

define void @vst4Qi8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4Qi8:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst4.8 {d16, d18, d20, d22}, [r0:256]!
;CHECK: vst4.8 {d17, d19, d21, d23}, [r0:256]
	%tmp1 = load <16 x i8>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v16i8(ptr %A, <16 x i8> %tmp1, <16 x i8> %tmp1, <16 x i8> %tmp1, <16 x i8> %tmp1, i32 64)
	ret void
}

define void @vst4Qi16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4Qi16:
;Check for no alignment specifier.
;CHECK: vst4.16 {d16, d18, d20, d22}, [r0]!
;CHECK: vst4.16 {d17, d19, d21, d23}, [r0]
	%tmp1 = load <8 x i16>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v8i16(ptr %A, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1)
	ret void
}

define void @vst4Qi32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4Qi32:
;CHECK: vst4.32
;CHECK: vst4.32
	%tmp1 = load <4 x i32>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v4i32(ptr %A, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 1)
	ret void
}

define void @vst4Qf(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vst4Qf:
;CHECK: vst4.32
;CHECK: vst4.32
	%tmp1 = load <4 x float>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v4f32(ptr %A, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1)
	ret void
}

;Check for a post-increment updating store.
define void @vst4Qf_update(ptr %ptr, ptr %B) nounwind {
;CHECK-LABEL: vst4Qf_update:
  ;CHECK: vst4.32 {d16, d18, d20, d22}, [r[[REG:[0-9]+]]]!
;CHECK: vst4.32 {d17, d19, d21, d23}, [r[[REG]]]!
	%A = load ptr, ptr %ptr
	%tmp1 = load <4 x float>, ptr %B
	call void @llvm.arm.neon.vst4.p0.v4f32(ptr %A, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1)
	%tmp2 = getelementptr float, ptr %A, i32 16
	store ptr %tmp2, ptr %ptr
	ret void
}

declare void @llvm.arm.neon.vst4.p0.v8i8(ptr, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0.v4i16(ptr, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0.v2i32(ptr, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0.v2f32(ptr, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0.v1i64(ptr, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32) nounwind

declare void @llvm.arm.neon.vst4.p0.v16i8(ptr, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0.v8i16(ptr, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0.v4i32(ptr, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0.v4f32(ptr, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32) nounwind
