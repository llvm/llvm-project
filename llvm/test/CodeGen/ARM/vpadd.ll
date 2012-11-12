; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vpaddi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vpaddi8:
;CHECK: vpadd.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vpaddi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vpaddi16:
;CHECK: vpadd.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpaddi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vpaddi32:
;CHECK: vpadd.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <2 x float> @vpaddf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vpaddf32:
;CHECK: vpadd.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vpadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float>, <2 x float>) nounwind readnone

define <4 x i16> @vpaddls8(<8 x i8>* %A) nounwind {
;CHECK: vpaddls8:
;CHECK: vpaddl.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vpaddls16(<4 x i16>* %A) nounwind {
;CHECK: vpaddls16:
;CHECK: vpaddl.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vpaddls.v2i32.v4i16(<4 x i16> %tmp1)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vpaddls32(<2 x i32>* %A) nounwind {
;CHECK: vpaddls32:
;CHECK: vpaddl.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vpaddls.v1i64.v2i32(<2 x i32> %tmp1)
	ret <1 x i64> %tmp2
}

define <4 x i16> @vpaddlu8(<8 x i8>* %A) nounwind {
;CHECK: vpaddlu8:
;CHECK: vpaddl.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vpaddlu.v4i16.v8i8(<8 x i8> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vpaddlu16(<4 x i16>* %A) nounwind {
;CHECK: vpaddlu16:
;CHECK: vpaddl.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16> %tmp1)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vpaddlu32(<2 x i32>* %A) nounwind {
;CHECK: vpaddlu32:
;CHECK: vpaddl.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vpaddlu.v1i64.v2i32(<2 x i32> %tmp1)
	ret <1 x i64> %tmp2
}

define <8 x i16> @vpaddlQs8(<16 x i8>* %A) nounwind {
;CHECK: vpaddlQs8:
;CHECK: vpaddl.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vpaddls.v8i16.v16i8(<16 x i8> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vpaddlQs16(<8 x i16>* %A) nounwind {
;CHECK: vpaddlQs16:
;CHECK: vpaddl.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vpaddls.v4i32.v8i16(<8 x i16> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vpaddlQs32(<4 x i32>* %A) nounwind {
;CHECK: vpaddlQs32:
;CHECK: vpaddl.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vpaddls.v2i64.v4i32(<4 x i32> %tmp1)
	ret <2 x i64> %tmp2
}

define <8 x i16> @vpaddlQu8(<16 x i8>* %A) nounwind {
;CHECK: vpaddlQu8:
;CHECK: vpaddl.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vpaddlQu16(<8 x i16>* %A) nounwind {
;CHECK: vpaddlQu16:
;CHECK: vpaddl.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vpaddlQu32(<4 x i32>* %A) nounwind {
;CHECK: vpaddlQu32:
;CHECK: vpaddl.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %tmp1)
	ret <2 x i64> %tmp2
}

; Test AddCombine optimization that generates a vpaddl.s
define void @addCombineToVPADDL() nounwind ssp {
; CHECK: vpaddl.s8
  %cbcr = alloca <16 x i8>, align 16
  %X = alloca <8 x i8>, align 8
  %tmp = load <16 x i8>* %cbcr
  %tmp1 = shufflevector <16 x i8> %tmp, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %tmp2 = load <16 x i8>* %cbcr
  %tmp3 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %add = add <8 x i8> %tmp3, %tmp1
  store <8 x i8> %add, <8 x i8>* %X, align 8
  ret void
}

declare <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpaddls.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpaddls.v1i64.v2i32(<2 x i32>) nounwind readnone

declare <4 x i16> @llvm.arm.neon.vpaddlu.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpaddlu.v1i64.v2i32(<2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vpaddls.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpaddls.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpaddls.v2i64.v4i32(<4 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32>) nounwind readnone
