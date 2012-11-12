; RUN: llc < %s -mtriple=armv6-eabi -mattr=+vfp2 -float-abi=hard
; PR4419

define float @__ieee754_acosf(float %x) nounwind {
entry:
	br i1 undef, label %bb, label %bb4

bb:		; preds = %entry
	ret float undef

bb4:		; preds = %entry
	br i1 undef, label %bb5, label %bb6

bb5:		; preds = %bb4
	ret float undef

bb6:		; preds = %bb4
	br i1 undef, label %bb11, label %bb12

bb11:		; preds = %bb6
	%0 = tail call float @__ieee754_sqrtf(float undef) nounwind		; <float> [#uses=1]
	%1 = fmul float %0, -2.000000e+00		; <float> [#uses=1]
	%2 = fadd float %1, 0x400921FB40000000		; <float> [#uses=1]
	ret float %2

bb12:		; preds = %bb6
	ret float undef
}

declare float @__ieee754_sqrtf(float)
