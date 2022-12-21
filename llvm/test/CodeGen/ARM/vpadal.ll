; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <4 x i16> @vpadals8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadals8:
;CHECK: vpadal.s8
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpadals.v4i16.v8i8(<4 x i16> %tmp1, <8 x i8> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpadals16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadals16:
;CHECK: vpadal.s16
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpadals.v2i32.v4i16(<2 x i32> %tmp1, <4 x i16> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vpadals32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadals32:
;CHECK: vpadal.s32
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vpadals.v1i64.v2i32(<1 x i64> %tmp1, <2 x i32> %tmp2)
	ret <1 x i64> %tmp3
}

define <4 x i16> @vpadalu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalu8:
;CHECK: vpadal.u8
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpadalu.v4i16.v8i8(<4 x i16> %tmp1, <8 x i8> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpadalu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalu16:
;CHECK: vpadal.u16
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpadalu.v2i32.v4i16(<2 x i32> %tmp1, <4 x i16> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vpadalu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalu32:
;CHECK: vpadal.u32
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vpadalu.v1i64.v2i32(<1 x i64> %tmp1, <2 x i32> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i16> @vpadalQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalQs8:
;CHECK: vpadal.s8
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vpadals.v8i16.v16i8(<8 x i16> %tmp1, <16 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vpadalQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalQs16:
;CHECK: vpadal.s16
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vpadals.v4i32.v8i16(<4 x i32> %tmp1, <8 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vpadalQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalQs32:
;CHECK: vpadal.s32
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vpadals.v2i64.v4i32(<2 x i64> %tmp1, <4 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vpadalQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalQu8:
;CHECK: vpadal.u8
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16> %tmp1, <16 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vpadalQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalQu16:
;CHECK: vpadal.u16
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vpadalu.v4i32.v8i16(<4 x i32> %tmp1, <8 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vpadalQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vpadalQu32:
;CHECK: vpadal.u32
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
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
