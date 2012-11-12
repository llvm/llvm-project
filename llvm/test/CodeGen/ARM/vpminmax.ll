; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vpmins8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vpmins8:
;CHECK: vpmin.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpmins.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vpmins16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vpmins16:
;CHECK: vpmin.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpmins.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpmins32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vpmins32:
;CHECK: vpmin.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpmins.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i8> @vpminu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vpminu8:
;CHECK: vpmin.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpminu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vpminu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vpminu16:
;CHECK: vpmin.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpminu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpminu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vpminu32:
;CHECK: vpmin.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpminu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <2 x float> @vpminf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vpminf32:
;CHECK: vpmin.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.arm.neon.vpmins.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vpmins.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpmins.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpmins.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vpminu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpminu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpminu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vpmins.v2f32(<2 x float>, <2 x float>) nounwind readnone

define <8 x i8> @vpmaxs8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vpmaxs8:
;CHECK: vpmax.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpmaxs.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vpmaxs16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vpmaxs16:
;CHECK: vpmax.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpmaxs.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpmaxs32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vpmaxs32:
;CHECK: vpmax.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpmaxs.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i8> @vpmaxu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vpmaxu8:
;CHECK: vpmax.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpmaxu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vpmaxu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vpmaxu16:
;CHECK: vpmax.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpmaxu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vpmaxu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vpmaxu32:
;CHECK: vpmax.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpmaxu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <2 x float> @vpmaxf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vpmaxf32:
;CHECK: vpmax.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.arm.neon.vpmaxs.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vpmaxs.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpmaxs.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpmaxs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vpmaxu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpmaxu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpmaxu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vpmaxs.v2f32(<2 x float>, <2 x float>) nounwind readnone
