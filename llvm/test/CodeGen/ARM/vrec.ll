; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <2 x i32> @vrecpei32(<2 x i32>* %A) nounwind {
;CHECK: vrecpei32:
;CHECK: vrecpe.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrecpe.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <4 x i32> @vrecpeQi32(<4 x i32>* %A) nounwind {
;CHECK: vrecpeQi32:
;CHECK: vrecpe.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrecpe.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x float> @vrecpef32(<2 x float>* %A) nounwind {
;CHECK: vrecpef32:
;CHECK: vrecpe.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = call <2 x float> @llvm.arm.neon.vrecpe.v2f32(<2 x float> %tmp1)
	ret <2 x float> %tmp2
}

define <4 x float> @vrecpeQf32(<4 x float>* %A) nounwind {
;CHECK: vrecpeQf32:
;CHECK: vrecpe.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float> %tmp1)
	ret <4 x float> %tmp2
}

declare <2 x i32> @llvm.arm.neon.vrecpe.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrecpe.v4i32(<4 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vrecpe.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float>) nounwind readnone

define <2 x float> @vrecpsf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vrecpsf32:
;CHECK: vrecps.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.arm.neon.vrecps.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <4 x float> @vrecpsQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vrecpsQf32:
;CHECK: vrecps.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = call <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

declare <2 x float> @llvm.arm.neon.vrecps.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <2 x i32> @vrsqrtei32(<2 x i32>* %A) nounwind {
;CHECK: vrsqrtei32:
;CHECK: vrsqrte.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrsqrte.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <4 x i32> @vrsqrteQi32(<4 x i32>* %A) nounwind {
;CHECK: vrsqrteQi32:
;CHECK: vrsqrte.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrsqrte.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x float> @vrsqrtef32(<2 x float>* %A) nounwind {
;CHECK: vrsqrtef32:
;CHECK: vrsqrte.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = call <2 x float> @llvm.arm.neon.vrsqrte.v2f32(<2 x float> %tmp1)
	ret <2 x float> %tmp2
}

define <4 x float> @vrsqrteQf32(<4 x float>* %A) nounwind {
;CHECK: vrsqrteQf32:
;CHECK: vrsqrte.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float> %tmp1)
	ret <4 x float> %tmp2
}

declare <2 x i32> @llvm.arm.neon.vrsqrte.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrsqrte.v4i32(<4 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vrsqrte.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float>) nounwind readnone

define <2 x float> @vrsqrtsf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vrsqrtsf32:
;CHECK: vrsqrts.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.arm.neon.vrsqrts.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <4 x float> @vrsqrtsQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vrsqrtsQf32:
;CHECK: vrsqrts.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = call <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

declare <2 x float> @llvm.arm.neon.vrsqrts.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float>, <4 x float>) nounwind readnone
