; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define void @vst4i8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vst4i8:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst4.8 {d16, d17, d18, d19}, [r0:64]
	%tmp1 = load <8 x i8>, <8 x i8>* %B
	call void @llvm.arm.neon.vst4.p0i8.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 8)
	ret void
}

;Check for a post-increment updating store with register increment.
define void @vst4i8_update(i8** %ptr, <8 x i8>* %B, i32 %inc) nounwind {
;CHECK-LABEL: vst4i8_update:
;CHECK: vst4.8 {d16, d17, d18, d19}, [r{{[0-9]+}}:128], r2
	%A = load i8*, i8** %ptr
	%tmp1 = load <8 x i8>, <8 x i8>* %B
	call void @llvm.arm.neon.vst4.p0i8.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 16)
	%tmp2 = getelementptr i8, i8* %A, i32 %inc
	store i8* %tmp2, i8** %ptr
	ret void
}

define void @vst4i16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vst4i16:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst4.16 {d16, d17, d18, d19}, [r0:128]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>, <4 x i16>* %B
	call void @llvm.arm.neon.vst4.p0i8.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 16)
	ret void
}

define void @vst4i32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vst4i32:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst4.32 {d16, d17, d18, d19}, [r0:256]
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>, <2 x i32>* %B
	call void @llvm.arm.neon.vst4.p0i8.v2i32(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 32)
	ret void
}

define void @vst4f(float* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vst4f:
;CHECK: vst4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>, <2 x float>* %B
	call void @llvm.arm.neon.vst4.p0i8.v2f32(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
	ret void
}

define void @vst4i64(i64* %A, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vst4i64:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst1.64 {d16, d17, d18, d19}, [r0:256]
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = load <1 x i64>, <1 x i64>* %B
	call void @llvm.arm.neon.vst4.p0i8.v1i64(i8* %tmp0, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 64)
	ret void
}

define void @vst4i64_update(i64** %ptr, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vst4i64_update:
;CHECK: vst1.64	{d16, d17, d18, d19}, [r{{[0-9]+}}]!
        %A = load i64*, i64** %ptr
        %tmp0 = bitcast i64* %A to i8*
        %tmp1 = load <1 x i64>, <1 x i64>* %B
        call void @llvm.arm.neon.vst4.p0i8.v1i64(i8* %tmp0, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 1)
        %tmp2 = getelementptr i64, i64* %A, i32 4
        store i64* %tmp2, i64** %ptr
        ret void
}

define void @vst4i64_reg_update(i64** %ptr, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vst4i64_reg_update:
;CHECK: vst1.64	{d16, d17, d18, d19}, [r{{[0-9]+}}], r{{[0-9]+}}
        %A = load i64*, i64** %ptr
        %tmp0 = bitcast i64* %A to i8*
        %tmp1 = load <1 x i64>, <1 x i64>* %B
        call void @llvm.arm.neon.vst4.p0i8.v1i64(i8* %tmp0, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 1)
        %tmp2 = getelementptr i64, i64* %A, i32 1
        store i64* %tmp2, i64** %ptr
        ret void
}

define void @vst4Qi8(i8* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vst4Qi8:
;Check the alignment value.  Max for this instruction is 256 bits:
;CHECK: vst4.8 {d16, d18, d20, d22}, [r0:256]!
;CHECK: vst4.8 {d17, d19, d21, d23}, [r0:256]
	%tmp1 = load <16 x i8>, <16 x i8>* %B
	call void @llvm.arm.neon.vst4.p0i8.v16i8(i8* %A, <16 x i8> %tmp1, <16 x i8> %tmp1, <16 x i8> %tmp1, <16 x i8> %tmp1, i32 64)
	ret void
}

define void @vst4Qi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vst4Qi16:
;Check for no alignment specifier.
;CHECK: vst4.16 {d16, d18, d20, d22}, [r0]!
;CHECK: vst4.16 {d17, d19, d21, d23}, [r0]
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>, <8 x i16>* %B
	call void @llvm.arm.neon.vst4.p0i8.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1)
	ret void
}

define void @vst4Qi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vst4Qi32:
;CHECK: vst4.32
;CHECK: vst4.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>, <4 x i32>* %B
	call void @llvm.arm.neon.vst4.p0i8.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 1)
	ret void
}

define void @vst4Qf(float* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vst4Qf:
;CHECK: vst4.32
;CHECK: vst4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>, <4 x float>* %B
	call void @llvm.arm.neon.vst4.p0i8.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1)
	ret void
}

;Check for a post-increment updating store.
define void @vst4Qf_update(float** %ptr, <4 x float>* %B) nounwind {
;CHECK-LABEL: vst4Qf_update:
  ;CHECK: vst4.32 {d16, d18, d20, d22}, [r[[REG:[0-9]+]]]!
;CHECK: vst4.32 {d17, d19, d21, d23}, [r[[REG]]]!
	%A = load float*, float** %ptr
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>, <4 x float>* %B
	call void @llvm.arm.neon.vst4.p0i8.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1)
	%tmp2 = getelementptr float, float* %A, i32 16
	store float* %tmp2, float** %ptr
	ret void
}

declare void @llvm.arm.neon.vst4.p0i8.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0i8.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0i8.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0i8.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0i8.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32) nounwind

declare void @llvm.arm.neon.vst4.p0i8.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0i8.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0i8.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst4.p0i8.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32) nounwind
