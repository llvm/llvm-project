; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vabas8(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabas8:
;CHECK: vaba.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = load <8 x i8>, ptr %C
	%tmp4 = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %tmp2, <8 x i8> %tmp3)
	%tmp5 = add <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vabas16(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabas16:
;CHECK: vaba.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = load <4 x i16>, ptr %C
	%tmp4 = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %tmp2, <4 x i16> %tmp3)
	%tmp5 = add <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

define <2 x i32> @vabas32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabas32:
;CHECK: vaba.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = load <2 x i32>, ptr %C
	%tmp4 = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %tmp2, <2 x i32> %tmp3)
	%tmp5 = add <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

define <8 x i8> @vabau8(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabau8:
;CHECK: vaba.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = load <8 x i8>, ptr %C
	%tmp4 = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %tmp2, <8 x i8> %tmp3)
	%tmp5 = add <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vabau16(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabau16:
;CHECK: vaba.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = load <4 x i16>, ptr %C
	%tmp4 = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %tmp2, <4 x i16> %tmp3)
	%tmp5 = add <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

define <2 x i32> @vabau32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabau32:
;CHECK: vaba.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = load <2 x i32>, ptr %C
	%tmp4 = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %tmp2, <2 x i32> %tmp3)
	%tmp5 = add <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

define <16 x i8> @vabaQs8(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabaQs8:
;CHECK: vaba.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = load <16 x i8>, ptr %C
	%tmp4 = call <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8> %tmp2, <16 x i8> %tmp3)
	%tmp5 = add <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

define <8 x i16> @vabaQs16(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabaQs16:
;CHECK: vaba.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = load <8 x i16>, ptr %C
	%tmp4 = call <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16> %tmp2, <8 x i16> %tmp3)
	%tmp5 = add <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vabaQs32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabaQs32:
;CHECK: vaba.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = load <4 x i32>, ptr %C
	%tmp4 = call <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32> %tmp2, <4 x i32> %tmp3)
	%tmp5 = add <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

define <16 x i8> @vabaQu8(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabaQu8:
;CHECK: vaba.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = load <16 x i8>, ptr %C
	%tmp4 = call <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8> %tmp2, <16 x i8> %tmp3)
	%tmp5 = add <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

define <8 x i16> @vabaQu16(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabaQu16:
;CHECK: vaba.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = load <8 x i16>, ptr %C
	%tmp4 = call <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16> %tmp2, <8 x i16> %tmp3)
	%tmp5 = add <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vabaQu32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabaQu32:
;CHECK: vaba.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = load <4 x i32>, ptr %C
	%tmp4 = call <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32> %tmp2, <4 x i32> %tmp3)
	%tmp5 = add <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

declare <8 x i8>  @llvm.arm.neon.vabds.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vabdu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @vabals8(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabals8:
;CHECK: vabal.s8
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = load <8 x i8>, ptr %C
	%tmp4 = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %tmp2, <8 x i8> %tmp3)
	%tmp5 = zext <8 x i8> %tmp4 to <8 x i16>
	%tmp6 = add <8 x i16> %tmp1, %tmp5
	ret <8 x i16> %tmp6
}

define <4 x i32> @vabals16(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabals16:
;CHECK: vabal.s16
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = load <4 x i16>, ptr %C
	%tmp4 = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %tmp2, <4 x i16> %tmp3)
	%tmp5 = zext <4 x i16> %tmp4 to <4 x i32>
	%tmp6 = add <4 x i32> %tmp1, %tmp5
	ret <4 x i32> %tmp6
}

define <2 x i64> @vabals32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabals32:
;CHECK: vabal.s32
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = load <2 x i32>, ptr %C
	%tmp4 = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %tmp2, <2 x i32> %tmp3)
	%tmp5 = zext <2 x i32> %tmp4 to <2 x i64>
	%tmp6 = add <2 x i64> %tmp1, %tmp5
	ret <2 x i64> %tmp6
}

define <8 x i16> @vabalu8(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabalu8:
;CHECK: vabal.u8
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = load <8 x i8>, ptr %C
	%tmp4 = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %tmp2, <8 x i8> %tmp3)
	%tmp5 = zext <8 x i8> %tmp4 to <8 x i16>
	%tmp6 = add <8 x i16> %tmp1, %tmp5
	ret <8 x i16> %tmp6
}

define <4 x i32> @vabalu16(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabalu16:
;CHECK: vabal.u16
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = load <4 x i16>, ptr %C
	%tmp4 = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %tmp2, <4 x i16> %tmp3)
	%tmp5 = zext <4 x i16> %tmp4 to <4 x i32>
	%tmp6 = add <4 x i32> %tmp1, %tmp5
	ret <4 x i32> %tmp6
}

define <2 x i64> @vabalu32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: vabalu32:
;CHECK: vabal.u32
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = load <2 x i32>, ptr %C
	%tmp4 = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %tmp2, <2 x i32> %tmp3)
	%tmp5 = zext <2 x i32> %tmp4 to <2 x i64>
	%tmp6 = add <2 x i64> %tmp1, %tmp5
	ret <2 x i64> %tmp6
}
