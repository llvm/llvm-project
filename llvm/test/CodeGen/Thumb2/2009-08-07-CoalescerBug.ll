; RUN: llc < %s -mtriple=armv7-eabi -mattr=+vfp2
; PR4686

	%a = type { i32 (...)** }
	%b = type { %a }
	%c = type { float, float, float, float }

declare arm_aapcs_vfpcc float @bar(%c*)

define arm_aapcs_vfpcc void @foo(%b* %x, %c* %y) {
entry:
	%0 = call arm_aapcs_vfpcc  float @bar(%c* %y)		; <float> [#uses=0]
	%1 = fadd float undef, undef		; <float> [#uses=1]
	store float %1, float* undef, align 8
	ret void
}
