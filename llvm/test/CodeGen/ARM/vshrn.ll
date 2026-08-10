; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vshrns8(ptr %A) nounwind {
;CHECK-LABEL: vshrns8:
;CHECK: vshrn.i16
	%tmp1 = load <8 x i16>, ptr %A
        %tmp2 = lshr <8 x i16> %tmp1, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
        %tmp3 = trunc <8 x i16> %tmp2 to <8 x i8>
	ret <8 x i8> %tmp3
}

define <4 x i16> @vshrns16(ptr %A) nounwind {
;CHECK-LABEL: vshrns16:
;CHECK: vshrn.i32
	%tmp1 = load <4 x i32>, ptr %A
        %tmp2 = ashr <4 x i32> %tmp1, <i32 16, i32 16, i32 16, i32 16>
        %tmp3 = trunc <4 x i32> %tmp2 to <4 x i16>
	ret <4 x i16> %tmp3
}

define <2 x i32> @vshrns32(ptr %A) nounwind {
;CHECK-LABEL: vshrns32:
;CHECK: vshrn.i64
	%tmp1 = load <2 x i64>, ptr %A
        %tmp2 = ashr <2 x i64> %tmp1, <i64 32, i64 32>
        %tmp3 = trunc <2 x i64> %tmp2 to <2 x i32>
	ret <2 x i32> %tmp3
}

define <8 x i8> @vshrns8_bad(ptr %A) nounwind {
; CHECK-LABEL: vshrns8_bad:
; CHECK: vshr.s16
; CHECK: vmovn.i16
        %tmp1 = load <8 x i16>, ptr %A
        %tmp2 = ashr <8 x i16> %tmp1, <i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9>
        %tmp3 = trunc <8 x i16> %tmp2 to <8 x i8>
        ret <8 x i8> %tmp3
}

define <4 x i16> @vshrns16_bad(ptr %A) nounwind {
; CHECK-LABEL: vshrns16_bad:
; CHECK: vshr.u32
; CHECK: vmovn.i32
        %tmp1 = load <4 x i32>, ptr %A
        %tmp2 = lshr <4 x i32> %tmp1, <i32 17, i32 17, i32 17, i32 17>
        %tmp3 = trunc <4 x i32> %tmp2 to <4 x i16>
        ret <4 x i16> %tmp3
}

define <2 x i32> @vshrns32_bad(ptr %A) nounwind {
; CHECK-LABEL: vshrns32_bad:
; CHECK: vshr.u64
; CHECK: vmovn.i64
        %tmp1 = load <2 x i64>, ptr %A
        %tmp2 = lshr <2 x i64> %tmp1, <i64 33, i64 33>
        %tmp3 = trunc <2 x i64> %tmp2 to <2 x i32>
        ret <2 x i32> %tmp3
}

define <8 x i8> @vrshrns8(ptr %A) nounwind {
;CHECK-LABEL: vrshrns8:
;CHECK: vrshrn.i16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vrshiftn.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vrshrns16(ptr %A) nounwind {
;CHECK-LABEL: vrshrns16:
;CHECK: vrshrn.i32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vrshiftn.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vrshrns32(ptr %A) nounwind {
;CHECK-LABEL: vrshrns32:
;CHECK: vrshrn.i64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrshiftn.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vrshiftn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshiftn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshiftn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone
