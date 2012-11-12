; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vqadds8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vqadds8:
;CHECK: vqadd.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqadds.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqadds16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vqadds16:
;CHECK: vqadd.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqadds.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqadds32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vqadds32:
;CHECK: vqadd.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqadds.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqadds64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vqadds64:
;CHECK: vqadd.s64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqadds.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vqaddu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vqaddu8:
;CHECK: vqadd.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqaddu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqaddu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vqaddu16:
;CHECK: vqadd.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqaddu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqaddu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vqaddu32:
;CHECK: vqadd.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqaddu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqaddu64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vqaddu64:
;CHECK: vqadd.u64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqaddu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vqaddQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vqaddQs8:
;CHECK: vqadd.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqadds.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqaddQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vqaddQs16:
;CHECK: vqadd.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqadds.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqaddQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vqaddQs32:
;CHECK: vqadd.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqaddQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vqaddQs64:
;CHECK: vqadd.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vqaddQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vqaddQu8:
;CHECK: vqadd.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqaddu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqaddQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vqaddQu16:
;CHECK: vqadd.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqaddu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqaddQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vqaddQu32:
;CHECK: vqadd.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqaddu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqaddQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vqaddQu64:
;CHECK: vqadd.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqaddu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vqadds.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqadds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqadds.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqadds.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqaddu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqaddu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqaddu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqaddu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqadds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqadds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqaddu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqaddu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqaddu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqaddu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
