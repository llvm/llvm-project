; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vhsubs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubs8:
;CHECK: vhsub.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vhsubs.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vhsubs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubs16:
;CHECK: vhsub.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vhsubs.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vhsubs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubs32:
;CHECK: vhsub.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vhsubs.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i8> @vhsubu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubu8:
;CHECK: vhsub.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vhsubu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vhsubu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubu16:
;CHECK: vhsub.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vhsubu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vhsubu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubu32:
;CHECK: vhsub.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vhsubu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <16 x i8> @vhsubQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubQs8:
;CHECK: vhsub.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vhsubs.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vhsubQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubQs16:
;CHECK: vhsub.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vhsubs.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vhsubQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubQs32:
;CHECK: vhsub.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vhsubs.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <16 x i8> @vhsubQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubQu8:
;CHECK: vhsub.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vhsubu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vhsubQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubQu16:
;CHECK: vhsub.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vhsubu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vhsubQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vhsubQu32:
;CHECK: vhsub.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vhsubu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vhsubs.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vhsubs.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vhsubs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vhsubu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vhsubu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vhsubu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vhsubs.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vhsubs.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vhsubs.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vhsubu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vhsubu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vhsubu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
