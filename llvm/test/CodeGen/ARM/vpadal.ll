; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <4 x i16> @vpadals8(<4 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK: vpadals8:
;CHECK: vpadal.s8
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpadals.v4i16.v8i8(<4 x i16> %tmp1, <8 x i8> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpadals16(<2 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK: vpadals16:
;CHECK: vpadal.s16
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpadals.v2i32.v4i16(<2 x i32> %tmp1, <4 x i16> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vpadals32(<1 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK: vpadals32:
;CHECK: vpadal.s32
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vpadals.v1i64.v2i32(<1 x i64> %tmp1, <2 x i32> %tmp2)
	ret <1 x i64> %tmp3
}

define <4 x i16> @vpadalu8(<4 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK: vpadalu8:
;CHECK: vpadal.u8
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpadalu.v4i16.v8i8(<4 x i16> %tmp1, <8 x i8> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpadalu16(<2 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK: vpadalu16:
;CHECK: vpadal.u16
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpadalu.v2i32.v4i16(<2 x i32> %tmp1, <4 x i16> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vpadalu32(<1 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK: vpadalu32:
;CHECK: vpadal.u32
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vpadalu.v1i64.v2i32(<1 x i64> %tmp1, <2 x i32> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i16> @vpadalQs8(<8 x i16>* %A, <16 x i8>* %B) nounwind {
;CHECK: vpadalQs8:
;CHECK: vpadal.s8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vpadals.v8i16.v16i8(<8 x i16> %tmp1, <16 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vpadalQs16(<4 x i32>* %A, <8 x i16>* %B) nounwind {
;CHECK: vpadalQs16:
;CHECK: vpadal.s16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vpadals.v4i32.v8i16(<4 x i32> %tmp1, <8 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vpadalQs32(<2 x i64>* %A, <4 x i32>* %B) nounwind {
;CHECK: vpadalQs32:
;CHECK: vpadal.s32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vpadals.v2i64.v4i32(<2 x i64> %tmp1, <4 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vpadalQu8(<8 x i16>* %A, <16 x i8>* %B) nounwind {
;CHECK: vpadalQu8:
;CHECK: vpadal.u8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16> %tmp1, <16 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vpadalQu16(<4 x i32>* %A, <8 x i16>* %B) nounwind {
;CHECK: vpadalQu16:
;CHECK: vpadal.u16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vpadalu.v4i32.v8i16(<4 x i32> %tmp1, <8 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vpadalQu32(<2 x i64>* %A, <4 x i32>* %B) nounwind {
;CHECK: vpadalQu32:
;CHECK: vpadal.u32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vpadalu.v2i64.v4i32(<2 x i64> %tmp1, <4 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

declare <4 x i16> @llvm.arm.neon.vpadals.v4i16.v8i8(<4 x i16>, <8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpadals.v2i32.v4i16(<2 x i32>, <4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpadals.v1i64.v2i32(<1 x i64>, <2 x i32>) nounwind readnone

declare <4 x i16> @llvm.arm.neon.vpadalu.v4i16.v8i8(<4 x i16>, <8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpadalu.v2i32.v4i16(<2 x i32>, <4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpadalu.v1i64.v2i32(<1 x i64>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vpadals.v8i16.v16i8(<8 x i16>, <16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpadals.v4i32.v8i16(<4 x i32>, <8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpadals.v2i64.v4i32(<2 x i64>, <4 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16>, <16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpadalu.v4i32.v8i16(<4 x i32>, <8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpadalu.v2i64.v4i32(<2 x i64>, <4 x i32>) nounwind readnone
