; RUN: llc < %s -mtriple=armv7-eabi -mattr=+vfp2
; PR4686

	%a = type { ptr }
	%b = type { %a }
	%c = type { float, float, float, float }

declare arm_aapcs_vfpcc float @bar(ptr)

define arm_aapcs_vfpcc void @foo(ptr %x, ptr %y) {
entry:
	%0 = call arm_aapcs_vfpcc  float @bar(ptr %y)		; <float> [#uses=0]
	%1 = fadd float undef, undef		; <float> [#uses=1]
	store float %1, ptr undef, align 8
	ret void
}
