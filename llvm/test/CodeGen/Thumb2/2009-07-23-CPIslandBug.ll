; RUN: llc < %s -mtriple=thumbv7-apple-darwin9 -mattr=+vfp2,+thumb2
; rdar://7083961

define i32 @value(i64 %b1, i64 %b2) nounwind readonly {
entry:
	%0 = icmp eq i32 undef, 0		; <i1> [#uses=1]
	%mod.0.ph.ph = select i1 %0, float -1.000000e+00, float 1.000000e+00		; <float> [#uses=1]
	br label %bb7

bb7:		; preds = %bb7, %entry
	br i1 undef, label %bb86.preheader, label %bb7

bb86.preheader:		; preds = %bb7
	%1 = fmul float %mod.0.ph.ph, 5.000000e+00		; <float> [#uses=0]
	br label %bb79

bb79:		; preds = %bb79, %bb86.preheader
	br i1 undef, label %bb119, label %bb79

bb119:		; preds = %bb79
	ret i32 undef
}
