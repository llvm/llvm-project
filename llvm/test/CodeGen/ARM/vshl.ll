; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vshls8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshls8:
;CHECK: vshl.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vshifts.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vshls16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshls16:
;CHECK: vshl.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vshifts.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vshls32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshls32:
;CHECK: vshl.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vshifts.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vshls64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshls64:
;CHECK: vshl.s64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <1 x i64>, ptr %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vshifts.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vshlu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlu8:
;CHECK: vshl.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vshlu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlu16:
;CHECK: vshl.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vshlu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlu32:
;CHECK: vshl.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vshlu64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlu64:
;CHECK: vshl.u64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <1 x i64>, ptr %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vshlQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlQs8:
;CHECK: vshl.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vshifts.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vshlQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlQs16:
;CHECK: vshl.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vshifts.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vshlQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlQs32:
;CHECK: vshl.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vshifts.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vshlQs64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlQs64:
;CHECK: vshl.s64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vshifts.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vshlQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlQu8:
;CHECK: vshl.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vshlQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlQu16:
;CHECK: vshl.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vshlQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlQu32:
;CHECK: vshl.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vshlQu64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vshlQu64:
;CHECK: vshl.u64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

; For left shifts by immediates, the signedness is irrelevant.
; Test a mix of both signed and unsigned intrinsics.

define <8 x i8> @vshli8(ptr %A) nounwind {
;CHECK-LABEL: vshli8:
;CHECK: vshl.i8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vshifts.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vshli16(ptr %A) nounwind {
;CHECK-LABEL: vshli16:
;CHECK: vshl.i16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vshli32(ptr %A) nounwind {
;CHECK-LABEL: vshli32:
;CHECK: vshl.i32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vshifts.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vshli64(ptr %A) nounwind {
;CHECK-LABEL: vshli64:
;CHECK: vshl.i64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 63 >)
	ret <1 x i64> %tmp2
}

define <16 x i8> @vshlQi8(ptr %A) nounwind {
;CHECK-LABEL: vshlQi8:
;CHECK: vshl.i8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vshifts.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vshlQi16(ptr %A) nounwind {
;CHECK-LABEL: vshlQi16:
;CHECK: vshl.i16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vshlQi32(ptr %A) nounwind {
;CHECK-LABEL: vshlQi32:
;CHECK: vshl.i32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vshifts.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 31, i32 31, i32 31, i32 31 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vshlQi64(ptr %A) nounwind {
;CHECK-LABEL: vshlQi64:
;CHECK: vshl.i64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 63, i64 63 >)
	ret <2 x i64> %tmp2
}

; Right shift by immediate:

define <8 x i8> @vshrs8(ptr %A) nounwind {
;CHECK-LABEL: vshrs8:
;CHECK: vshr.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vshifts.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vshrs16(ptr %A) nounwind {
;CHECK-LABEL: vshrs16:
;CHECK: vshr.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vshifts.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vshrs32(ptr %A) nounwind {
;CHECK-LABEL: vshrs32:
;CHECK: vshr.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vshifts.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 -32, i32 -32 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vshrs64(ptr %A) nounwind {
;CHECK-LABEL: vshrs64:
;CHECK: vshr.s64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vshifts.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 -64 >)
	ret <1 x i64> %tmp2
}

define <8 x i8> @vshru8(ptr %A) nounwind {
;CHECK-LABEL: vshru8:
;CHECK: vshr.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vshru16(ptr %A) nounwind {
;CHECK-LABEL: vshru16:
;CHECK: vshr.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vshru32(ptr %A) nounwind {
;CHECK-LABEL: vshru32:
;CHECK: vshr.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 -32, i32 -32 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vshru64(ptr %A) nounwind {
;CHECK-LABEL: vshru64:
;CHECK: vshr.u64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 -64 >)
	ret <1 x i64> %tmp2
}

define <16 x i8> @vshrQs8(ptr %A) nounwind {
;CHECK-LABEL: vshrQs8:
;CHECK: vshr.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vshifts.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vshrQs16(ptr %A) nounwind {
;CHECK-LABEL: vshrQs16:
;CHECK: vshr.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vshifts.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vshrQs32(ptr %A) nounwind {
;CHECK-LABEL: vshrQs32:
;CHECK: vshr.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vshifts.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vshrQs64(ptr %A) nounwind {
;CHECK-LABEL: vshrQs64:
;CHECK: vshr.s64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vshifts.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 -64, i64 -64 >)
	ret <2 x i64> %tmp2
}

define <16 x i8> @vshrQu8(ptr %A) nounwind {
;CHECK-LABEL: vshrQu8:
;CHECK: vshr.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vshrQu16(ptr %A) nounwind {
;CHECK-LABEL: vshrQu16:
;CHECK: vshr.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vshrQu32(ptr %A) nounwind {
;CHECK-LABEL: vshrQu32:
;CHECK: vshr.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vshrQu64(ptr %A) nounwind {
;CHECK-LABEL: vshrQu64:
;CHECK: vshr.u64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 -64, i64 -64 >)
	ret <2 x i64> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @vrshls8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshls8:
;CHECK: vrshl.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrshifts.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vrshls16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshls16:
;CHECK: vrshl.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vrshls32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshls32:
;CHECK: vrshl.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vrshls64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshls64:
;CHECK: vrshl.s64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <1 x i64>, ptr %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vrshlu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlu8:
;CHECK: vrshl.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vrshlu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlu16:
;CHECK: vrshl.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vrshlu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlu32:
;CHECK: vrshl.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vrshlu64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlu64:
;CHECK: vrshl.u64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <1 x i64>, ptr %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vrshlQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlQs8:
;CHECK: vrshl.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vrshlQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlQs16:
;CHECK: vrshl.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vrshlQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlQs32:
;CHECK: vrshl.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vrshlQs64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlQs64:
;CHECK: vrshl.s64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vrshlQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlQu8:
;CHECK: vrshl.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vrshlQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlQu16:
;CHECK: vrshl.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vrshlQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlQu32:
;CHECK: vrshl.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vrshlQu64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vrshlQu64:
;CHECK: vrshl.u64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i8> @vrshrs8(ptr %A) nounwind {
;CHECK-LABEL: vrshrs8:
;CHECK: vrshr.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vrshifts.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vrshrs16(ptr %A) nounwind {
;CHECK-LABEL: vrshrs16:
;CHECK: vrshr.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vrshrs32(ptr %A) nounwind {
;CHECK-LABEL: vrshrs32:
;CHECK: vrshr.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 -32, i32 -32 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vrshrs64(ptr %A) nounwind {
;CHECK-LABEL: vrshrs64:
;CHECK: vrshr.s64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 -64 >)
	ret <1 x i64> %tmp2
}

define <8 x i8> @vrshru8(ptr %A) nounwind {
;CHECK-LABEL: vrshru8:
;CHECK: vrshr.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vrshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vrshru16(ptr %A) nounwind {
;CHECK-LABEL: vrshru16:
;CHECK: vrshr.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vrshru32(ptr %A) nounwind {
;CHECK-LABEL: vrshru32:
;CHECK: vrshr.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 -32, i32 -32 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vrshru64(ptr %A) nounwind {
;CHECK-LABEL: vrshru64:
;CHECK: vrshr.u64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 -64 >)
	ret <1 x i64> %tmp2
}

define <16 x i8> @vrshrQs8(ptr %A) nounwind {
;CHECK-LABEL: vrshrQs8:
;CHECK: vrshr.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vrshrQs16(ptr %A) nounwind {
;CHECK-LABEL: vrshrQs16:
;CHECK: vrshr.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vrshrQs32(ptr %A) nounwind {
;CHECK-LABEL: vrshrQs32:
;CHECK: vrshr.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vrshrQs64(ptr %A) nounwind {
;CHECK-LABEL: vrshrQs64:
;CHECK: vrshr.s64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 -64, i64 -64 >)
	ret <2 x i64> %tmp2
}

define <16 x i8> @vrshrQu8(ptr %A) nounwind {
;CHECK-LABEL: vrshrQu8:
;CHECK: vrshr.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vrshrQu16(ptr %A) nounwind {
;CHECK-LABEL: vrshrQu16:
;CHECK: vrshr.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vrshrQu32(ptr %A) nounwind {
;CHECK-LABEL: vrshrQu32:
;CHECK: vrshr.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vrshrQu64(ptr %A) nounwind {
;CHECK-LABEL: vrshrQu64:
;CHECK: vrshr.u64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 -64, i64 -64 >)
	ret <2 x i64> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vrshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vrshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
