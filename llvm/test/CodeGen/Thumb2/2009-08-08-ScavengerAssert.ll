; RUN: llc < %s -mtriple=armv7-eabi -mattr=+vfp2
; PR4686

@g_d = external global double		; <ptr> [#uses=1]

define void @foo(float %yIncr) {
entry:
	br i1 undef, label %bb, label %bb4

bb:		; preds = %entry
	%0 = call arm_aapcs_vfpcc  float @bar()		; <float> [#uses=1]
	%1 = fpext float %0 to double		; <double> [#uses=1]
	store double %1, ptr @g_d, align 8
	br label %bb4

bb4:		; preds = %bb, %entry
	unreachable
}

declare arm_aapcs_vfpcc float @bar()
