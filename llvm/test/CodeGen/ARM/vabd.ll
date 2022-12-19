; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vabds8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabds8:
;CHECK: vabd.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vabds16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabds16:
;CHECK: vabd.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vabds32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabds32:
;CHECK: vabd.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i8> @vabdu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdu8:
;CHECK: vabd.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vabdu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdu16:
;CHECK: vabd.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vabdu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdu32:
;CHECK: vabd.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <2 x float> @vabdf32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdf32:
;CHECK: vabd.f32
	%tmp1 = load <2 x float>, ptr %A
	%tmp2 = load <2 x float>, ptr %B
	%tmp3 = call <2 x float> @llvm.arm.neon.vabds.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <16 x i8> @vabdQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdQs8:
;CHECK: vabd.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vabdQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdQs16:
;CHECK: vabd.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vabdQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdQs32:
;CHECK: vabd.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <16 x i8> @vabdQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdQu8:
;CHECK: vabd.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vabdQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdQu16:
;CHECK: vabd.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vabdQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdQu32:
;CHECK: vabd.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <4 x float> @vabdQf32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdQf32:
;CHECK: vabd.f32
	%tmp1 = load <4 x float>, ptr %A
	%tmp2 = load <4 x float>, ptr %B
	%tmp3 = call <4 x float> @llvm.arm.neon.vabds.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vabds.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vabdu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vabds.v2f32(<2 x float>, <2 x float>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <4 x float> @llvm.arm.neon.vabds.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <8 x i16> @vabdls8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdls8:
;CHECK: vabdl.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	%tmp4 = zext <8 x i8> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vabdls16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdls16:
;CHECK: vabdl.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	%tmp4 = zext <4 x i16> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @vabdls32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdls32:
;CHECK: vabdl.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	%tmp4 = zext <2 x i32> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i16> @vabdlu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdlu8:
;CHECK: vabdl.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	%tmp4 = zext <8 x i8> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vabdlu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdlu16:
;CHECK: vabdl.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	%tmp4 = zext <4 x i16> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @vabdlu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vabdlu32:
;CHECK: vabdl.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	%tmp4 = zext <2 x i32> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}
