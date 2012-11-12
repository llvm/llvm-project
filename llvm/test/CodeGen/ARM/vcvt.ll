; RUN: llc < %s -march=arm -mattr=+neon,+fp16 | FileCheck %s

define <2 x i32> @vcvt_f32tos32(<2 x float>* %A) nounwind {
;CHECK: vcvt_f32tos32:
;CHECK: vcvt.s32.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = fptosi <2 x float> %tmp1 to <2 x i32>
	ret <2 x i32> %tmp2
}

define <2 x i32> @vcvt_f32tou32(<2 x float>* %A) nounwind {
;CHECK: vcvt_f32tou32:
;CHECK: vcvt.u32.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = fptoui <2 x float> %tmp1 to <2 x i32>
	ret <2 x i32> %tmp2
}

define <2 x float> @vcvt_s32tof32(<2 x i32>* %A) nounwind {
;CHECK: vcvt_s32tof32:
;CHECK: vcvt.f32.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = sitofp <2 x i32> %tmp1 to <2 x float>
	ret <2 x float> %tmp2
}

define <2 x float> @vcvt_u32tof32(<2 x i32>* %A) nounwind {
;CHECK: vcvt_u32tof32:
;CHECK: vcvt.f32.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = uitofp <2 x i32> %tmp1 to <2 x float>
	ret <2 x float> %tmp2
}

define <4 x i32> @vcvtQ_f32tos32(<4 x float>* %A) nounwind {
;CHECK: vcvtQ_f32tos32:
;CHECK: vcvt.s32.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = fptosi <4 x float> %tmp1 to <4 x i32>
	ret <4 x i32> %tmp2
}

define <4 x i32> @vcvtQ_f32tou32(<4 x float>* %A) nounwind {
;CHECK: vcvtQ_f32tou32:
;CHECK: vcvt.u32.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = fptoui <4 x float> %tmp1 to <4 x i32>
	ret <4 x i32> %tmp2
}

define <4 x float> @vcvtQ_s32tof32(<4 x i32>* %A) nounwind {
;CHECK: vcvtQ_s32tof32:
;CHECK: vcvt.f32.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = sitofp <4 x i32> %tmp1 to <4 x float>
	ret <4 x float> %tmp2
}

define <4 x float> @vcvtQ_u32tof32(<4 x i32>* %A) nounwind {
;CHECK: vcvtQ_u32tof32:
;CHECK: vcvt.f32.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = uitofp <4 x i32> %tmp1 to <4 x float>
	ret <4 x float> %tmp2
}

define <2 x i32> @vcvt_n_f32tos32(<2 x float>* %A) nounwind {
;CHECK: vcvt_n_f32tos32:
;CHECK: vcvt.s32.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float> %tmp1, i32 1)
	ret <2 x i32> %tmp2
}

define <2 x i32> @vcvt_n_f32tou32(<2 x float>* %A) nounwind {
;CHECK: vcvt_n_f32tou32:
;CHECK: vcvt.u32.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float> %tmp1, i32 1)
	ret <2 x i32> %tmp2
}

define <2 x float> @vcvt_n_s32tof32(<2 x i32>* %A) nounwind {
;CHECK: vcvt_n_s32tof32:
;CHECK: vcvt.f32.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x float> @llvm.arm.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32> %tmp1, i32 1)
	ret <2 x float> %tmp2
}

define <2 x float> @vcvt_n_u32tof32(<2 x i32>* %A) nounwind {
;CHECK: vcvt_n_u32tof32:
;CHECK: vcvt.f32.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x float> @llvm.arm.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32> %tmp1, i32 1)
	ret <2 x float> %tmp2
}

declare <2 x i32> @llvm.arm.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float>, i32) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float>, i32) nounwind readnone
declare <2 x float> @llvm.arm.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32>, i32) nounwind readnone
declare <2 x float> @llvm.arm.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32>, i32) nounwind readnone

define <4 x i32> @vcvtQ_n_f32tos32(<4 x float>* %A) nounwind {
;CHECK: vcvtQ_n_f32tos32:
;CHECK: vcvt.s32.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float> %tmp1, i32 1)
	ret <4 x i32> %tmp2
}

define <4 x i32> @vcvtQ_n_f32tou32(<4 x float>* %A) nounwind {
;CHECK: vcvtQ_n_f32tou32:
;CHECK: vcvt.u32.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float> %tmp1, i32 1)
	ret <4 x i32> %tmp2
}

define <4 x float> @vcvtQ_n_s32tof32(<4 x i32>* %A) nounwind {
;CHECK: vcvtQ_n_s32tof32:
;CHECK: vcvt.f32.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32> %tmp1, i32 1)
	ret <4 x float> %tmp2
}

define <4 x float> @vcvtQ_n_u32tof32(<4 x i32>* %A) nounwind {
;CHECK: vcvtQ_n_u32tof32:
;CHECK: vcvt.f32.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> %tmp1, i32 1)
	ret <4 x float> %tmp2
}

declare <4 x i32> @llvm.arm.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float>, i32) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float>, i32) nounwind readnone
declare <4 x float> @llvm.arm.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32>, i32) nounwind readnone
declare <4 x float> @llvm.arm.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32>, i32) nounwind readnone

define <4 x float> @vcvt_f16tof32(<4 x i16>* %A) nounwind {
;CHECK: vcvt_f16tof32:
;CHECK: vcvt.f32.f16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16> %tmp1)
	ret <4 x float> %tmp2
}

define <4 x i16> @vcvt_f32tof16(<4 x float>* %A) nounwind {
;CHECK: vcvt_f32tof16:
;CHECK: vcvt.f16.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float> %tmp1)
	ret <4 x i16> %tmp2
}

declare <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float>) nounwind readnone
