; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vabss8(ptr %A) nounwind {
;CHECK-LABEL: vabss8:
;CHECK: vabs.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vabs.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <8 x i8> @vabss8_fold(ptr %A) nounwind {
; CHECK-LABEL: vabss8_fold:
; CHECK:       vldr d16, .LCPI1_0
; CHECK:       .LCPI1_0:
; CHECK-NEXT:    .byte 128 @ 0x80
; CHECK-NEXT:    .byte 127 @ 0x7f
; CHECK-NEXT:    .byte 1 @ 0x1
; CHECK-NEXT:    .byte 0 @ 0x0
; CHECK-NEXT:    .byte 1 @ 0x1
; CHECK-NEXT:    .byte 127 @ 0x7f
; CHECK-NEXT:    .byte 128 @ 0x80
; CHECK-NEXT:    .byte 1 @ 0x1
	%tmp1 = call <8 x i8> @llvm.arm.neon.vabs.v8i8(<8 x i8> <i8 -128, i8 -127, i8 -1, i8 0, i8 1, i8 127, i8 128, i8 255>)
	ret <8 x i8> %tmp1
}

define <4 x i16> @vabss16(ptr %A) nounwind {
;CHECK-LABEL: vabss16:
;CHECK: vabs.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vabs.v4i16(<4 x i16> %tmp1)
	ret <4 x i16> %tmp2
}

define <4 x i16> @vabss16_fold() nounwind {
; CHECK-LABEL: vabss16_fold:
; CHECK:       vldr d16, .LCPI3_0
; CHECK:       .LCPI3_0:
; CHECK-NEXT:    .short 32768 @ 0x8000
; CHECK-NEXT:    .short 32767 @ 0x7fff
; CHECK-NEXT:    .short 255 @ 0xff
; CHECK-NEXT:    .short 32768 @ 0x8000
	%tmp1 = call <4 x i16> @llvm.arm.neon.vabs.v4i16(<4 x i16> <i16 -32768, i16 -32767, i16 255, i16 32768>)
	ret <4 x i16> %tmp1
}

define <2 x i32> @vabss32(ptr %A) nounwind {
;CHECK-LABEL: vabss32:
;CHECK: vabs.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vabs.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <2 x i32> @vabss32_fold() nounwind {
; CHECK-LABEL: vabss32_fold:
; CHECK:       vldr d16, .LCPI5_0
; CHECK:       .LCPI5_0:
; CHECK-NEXT:    .long 2147483647 @ 0x7fffffff
; CHECK-NEXT:    .long 2147483648 @ 0x80000000
	%tmp1 = call <2 x i32> @llvm.arm.neon.vabs.v2i32(<2 x i32> <i32 -2147483647, i32 2147483648>)
	ret <2 x i32> %tmp1
}

define <2 x float> @vabsf32(ptr %A) nounwind {
;CHECK-LABEL: vabsf32:
;CHECK: vabs.f32
	%tmp1 = load <2 x float>, ptr %A
	%tmp2 = call <2 x float> @llvm.fabs.v2f32(<2 x float> %tmp1)
	ret <2 x float> %tmp2
}

define <16 x i8> @vabsQs8(ptr %A) nounwind {
;CHECK-LABEL: vabsQs8:
;CHECK: vabs.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vabs.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vabsQs16(ptr %A) nounwind {
;CHECK-LABEL: vabsQs16:
;CHECK: vabs.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vabsQs32(ptr %A) nounwind {
;CHECK-LABEL: vabsQs32:
;CHECK: vabs.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vabs.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

define <4 x float> @vabsQf32(ptr %A) nounwind {
;CHECK-LABEL: vabsQf32:
;CHECK: vabs.f32
	%tmp1 = load <4 x float>, ptr %A
	%tmp2 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %tmp1)
	ret <4 x float> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vabs.v8i8(<8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vabs.v4i16(<4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vabs.v2i32(<2 x i32>) nounwind readnone
declare <2 x float> @llvm.fabs.v2f32(<2 x float>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vabs.v16i8(<16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabs.v4i32(<4 x i32>) nounwind readnone
declare <4 x float> @llvm.fabs.v4f32(<4 x float>) nounwind readnone

define <8 x i8> @vqabss8(ptr %A) nounwind {
;CHECK-LABEL: vqabss8:
;CHECK: vqabs.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqabs.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqabss16(ptr %A) nounwind {
;CHECK-LABEL: vqabss16:
;CHECK: vqabs.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqabs.v4i16(<4 x i16> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqabss32(ptr %A) nounwind {
;CHECK-LABEL: vqabss32:
;CHECK: vqabs.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqabs.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <16 x i8> @vqabsQs8(ptr %A) nounwind {
;CHECK-LABEL: vqabsQs8:
;CHECK: vqabs.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vqabs.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vqabsQs16(ptr %A) nounwind {
;CHECK-LABEL: vqabsQs16:
;CHECK: vqabs.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vqabs.v8i16(<8 x i16> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vqabsQs32(ptr %A) nounwind {
;CHECK-LABEL: vqabsQs32:
;CHECK: vqabs.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vqabs.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqabs.v8i8(<8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqabs.v4i16(<4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqabs.v2i32(<2 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqabs.v16i8(<16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqabs.v8i16(<8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqabs.v4i32(<4 x i32>) nounwind readnone
