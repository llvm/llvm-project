; RUN: llc < %s -mtriple=i686-- -mattr=+sse2

define i64 @__divsc3(float %a, float %b, float %c, float %d) nounwind readnone  {
entry:
	br i1 false, label %bb56, label %bb33

bb33:		; preds = %entry
	br label %bb56

bb56:		; preds = %bb33, %entry
	%tmp36.pn = phi float [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %entry ]		; <float> [#uses=1]
	%b.pn509 = phi float [ %b, %bb33 ], [ %a, %entry ]		; <float> [#uses=1]
	%a.pn = phi float [ %a, %bb33 ], [ %b, %entry ]		; <float> [#uses=1]
	%tmp41.pn508 = phi float [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %entry ]		; <float> [#uses=1]
	%tmp51.pn = phi float [ 0.000000e+00, %bb33 ], [ %a, %entry ]		; <float> [#uses=1]
	%tmp44.pn = fmul float %tmp36.pn, %b.pn509		; <float> [#uses=1]
	%tmp46.pn = fadd float %tmp44.pn, %a.pn		; <float> [#uses=1]
	%tmp53.pn = fsub float 0.000000e+00, %tmp51.pn		; <float> [#uses=1]
	%x.0 = fdiv float %tmp46.pn, %tmp41.pn508		; <float> [#uses=1]
	%y.0 = fdiv float %tmp53.pn, 0.000000e+00		; <float> [#uses=1]
	br i1 false, label %bb433, label %bb98

bb98:		; preds = %bb56
	%tmp102 = fmul float 0.000000e+00, %a		; <float> [#uses=1]
	%tmp106 = fmul float 0.000000e+00, %b		; <float> [#uses=1]
	br label %bb433

bb433:		; preds = %bb98, %bb56
	%x.1 = phi float [ %tmp102, %bb98 ], [ %x.0, %bb56 ]		; <float> [#uses=0]
	%y.1 = phi float [ %tmp106, %bb98 ], [ %y.0, %bb56 ]		; <float> [#uses=1]
	%tmp460 = fadd float %y.1, 0.000000e+00		; <float> [#uses=0]
	ret i64 0
}
