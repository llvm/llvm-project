; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vqshls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vqshls8:
;CHECK: vqshl.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqshls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vqshls16:
;CHECK: vqshl.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqshls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vqshls32:
;CHECK: vqshl.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqshls64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vqshls64:
;CHECK: vqshl.s64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vqshlu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vqshlu8:
;CHECK: vqshl.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqshlu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vqshlu16:
;CHECK: vqshl.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqshlu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vqshlu32:
;CHECK: vqshl.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqshlu64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vqshlu64:
;CHECK: vqshl.u64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vqshlQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vqshlQs8:
;CHECK: vqshl.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqshlQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vqshlQs16:
;CHECK: vqshl.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqshlQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vqshlQs32:
;CHECK: vqshl.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqshlQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vqshlQs64:
;CHECK: vqshl.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vqshlQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vqshlQu8:
;CHECK: vqshl.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqshlQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vqshlQu16:
;CHECK: vqshl.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqshlQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vqshlQu32:
;CHECK: vqshl.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqshlQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vqshlQu64:
;CHECK: vqshl.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i8> @vqshls_n8(<8 x i8>* %A) nounwind {
;CHECK: vqshls_n8:
;CHECK: vqshl.s8{{.*#7}}
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshls_n16(<4 x i16>* %A) nounwind {
;CHECK: vqshls_n16:
;CHECK: vqshl.s16{{.*#15}}
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshls_n32(<2 x i32>* %A) nounwind {
;CHECK: vqshls_n32:
;CHECK: vqshl.s32{{.*#31}}
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vqshls_n64(<1 x i64>* %A) nounwind {
;CHECK: vqshls_n64:
;CHECK: vqshl.s64{{.*#63}}
	%tmp1 = load <1 x i64>* %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 63 >)
	ret <1 x i64> %tmp2
}

define <8 x i8> @vqshlu_n8(<8 x i8>* %A) nounwind {
;CHECK: vqshlu_n8:
;CHECK: vqshl.u8{{.*#7}}
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshlu_n16(<4 x i16>* %A) nounwind {
;CHECK: vqshlu_n16:
;CHECK: vqshl.u16{{.*#15}}
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshlu_n32(<2 x i32>* %A) nounwind {
;CHECK: vqshlu_n32:
;CHECK: vqshl.u32{{.*#31}}
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vqshlu_n64(<1 x i64>* %A) nounwind {
;CHECK: vqshlu_n64:
;CHECK: vqshl.u64{{.*#63}}
	%tmp1 = load <1 x i64>* %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 63 >)
	ret <1 x i64> %tmp2
}

define <8 x i8> @vqshlsu_n8(<8 x i8>* %A) nounwind {
;CHECK: vqshlsu_n8:
;CHECK: vqshlu.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshiftsu.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshlsu_n16(<4 x i16>* %A) nounwind {
;CHECK: vqshlsu_n16:
;CHECK: vqshlu.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshiftsu.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshlsu_n32(<2 x i32>* %A) nounwind {
;CHECK: vqshlsu_n32:
;CHECK: vqshlu.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshiftsu.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vqshlsu_n64(<1 x i64>* %A) nounwind {
;CHECK: vqshlsu_n64:
;CHECK: vqshlu.s64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vqshiftsu.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 63 >)
	ret <1 x i64> %tmp2
}

define <16 x i8> @vqshlQs_n8(<16 x i8>* %A) nounwind {
;CHECK: vqshlQs_n8:
;CHECK: vqshl.s8{{.*#7}}
	%tmp1 = load <16 x i8>* %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vqshlQs_n16(<8 x i16>* %A) nounwind {
;CHECK: vqshlQs_n16:
;CHECK: vqshl.s16{{.*#15}}
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vqshlQs_n32(<4 x i32>* %A) nounwind {
;CHECK: vqshlQs_n32:
;CHECK: vqshl.s32{{.*#31}}
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 31, i32 31, i32 31, i32 31 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vqshlQs_n64(<2 x i64>* %A) nounwind {
;CHECK: vqshlQs_n64:
;CHECK: vqshl.s64{{.*#63}}
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 63, i64 63 >)
	ret <2 x i64> %tmp2
}

define <16 x i8> @vqshlQu_n8(<16 x i8>* %A) nounwind {
;CHECK: vqshlQu_n8:
;CHECK: vqshl.u8{{.*#7}}
	%tmp1 = load <16 x i8>* %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vqshlQu_n16(<8 x i16>* %A) nounwind {
;CHECK: vqshlQu_n16:
;CHECK: vqshl.u16{{.*#15}}
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vqshlQu_n32(<4 x i32>* %A) nounwind {
;CHECK: vqshlQu_n32:
;CHECK: vqshl.u32{{.*#31}}
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 31, i32 31, i32 31, i32 31 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vqshlQu_n64(<2 x i64>* %A) nounwind {
;CHECK: vqshlQu_n64:
;CHECK: vqshl.u64{{.*#63}}
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 63, i64 63 >)
	ret <2 x i64> %tmp2
}

define <16 x i8> @vqshlQsu_n8(<16 x i8>* %A) nounwind {
;CHECK: vqshlQsu_n8:
;CHECK: vqshlu.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vqshiftsu.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vqshlQsu_n16(<8 x i16>* %A) nounwind {
;CHECK: vqshlQsu_n16:
;CHECK: vqshlu.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vqshiftsu.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vqshlQsu_n32(<4 x i32>* %A) nounwind {
;CHECK: vqshlQsu_n32:
;CHECK: vqshlu.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vqshiftsu.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 31, i32 31, i32 31, i32 31 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vqshlQsu_n64(<2 x i64>* %A) nounwind {
;CHECK: vqshlQsu_n64:
;CHECK: vqshlu.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vqshiftsu.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 63, i64 63 >)
	ret <2 x i64> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqshiftsu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshiftsu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshiftsu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqshiftsu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqshiftsu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqshiftsu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqshiftsu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqshiftsu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @vqrshls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vqrshls8:
;CHECK: vqrshl.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqrshifts.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqrshls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vqrshls16:
;CHECK: vqrshl.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqrshifts.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqrshls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vqrshls32:
;CHECK: vqrshl.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqrshifts.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqrshls64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vqrshls64:
;CHECK: vqrshl.s64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqrshifts.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vqrshlu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vqrshlu8:
;CHECK: vqrshl.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqrshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqrshlu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vqrshlu16:
;CHECK: vqrshl.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqrshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqrshlu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vqrshlu32:
;CHECK: vqrshl.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqrshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqrshlu64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vqrshlu64:
;CHECK: vqrshl.u64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqrshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vqrshlQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vqrshlQs8:
;CHECK: vqrshl.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqrshifts.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqrshlQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vqrshlQs16:
;CHECK: vqrshl.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqrshifts.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqrshlQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vqrshlQs32:
;CHECK: vqrshl.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqrshifts.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqrshlQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vqrshlQs64:
;CHECK: vqrshl.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqrshifts.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vqrshlQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vqrshlQu8:
;CHECK: vqrshl.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqrshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqrshlQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vqrshlQu16:
;CHECK: vqrshl.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqrshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqrshlQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vqrshlQu32:
;CHECK: vqrshl.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqrshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqrshlQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vqrshlQu64:
;CHECK: vqrshl.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqrshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vqrshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqrshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqrshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqrshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqrshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqrshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqrshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqrshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqrshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqrshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqrshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqrshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqrshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqrshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqrshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqrshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
