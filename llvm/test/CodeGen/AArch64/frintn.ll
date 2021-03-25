; RUN: llc -mtriple=aarch64-eabi -mattr=+fullfp16 %s -o - | FileCheck %s

; The llvm.aarch64.neon.frintn intrinsic should be auto-upgraded to the
; target-independent roundeven intrinsic.

define <4 x half> @frintn_4h(<4 x half> %A) nounwind {
;CHECK-LABEL: frintn_4h:
;CHECK: frintn v0.4h, v0.4h
;CHECK-NEXT: ret
	%tmp3 = call <4 x half> @llvm.aarch64.neon.frintn.v4f16(<4 x half> %A)
	ret <4 x half> %tmp3
}

define <2 x float> @frintn_2s(<2 x float> %A) nounwind {
;CHECK-LABEL: frintn_2s:
;CHECK: frintn v0.2s, v0.2s
;CHECK-NEXT: ret
	%tmp3 = call <2 x float> @llvm.aarch64.neon.frintn.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frintn_4s(<4 x float> %A) nounwind {
;CHECK-LABEL: frintn_4s:
;CHECK: frintn v0.4s, v0.4s
;CHECK-NEXT: ret
	%tmp3 = call <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @frintn_2d(<2 x double> %A) nounwind {
;CHECK-LABEL: frintn_2d:
;CHECK: frintn v0.2d, v0.2d
;CHECK-NEXT: ret
	%tmp3 = call <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <4 x half> @llvm.aarch64.neon.frintn.v4f16(<4 x half>) nounwind readnone
declare <2 x float> @llvm.aarch64.neon.frintn.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double>) nounwind readnone
