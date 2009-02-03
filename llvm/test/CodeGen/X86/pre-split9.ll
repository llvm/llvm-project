; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -pre-alloc-split -stats |& \
; RUN:   grep {pre-alloc-split} | grep {Number of intervals split} | grep 1

@current_surfaces.b = external global i1		; <i1*> [#uses=1]

declare double @sin(double) nounwind readonly

declare double @asin(double) nounwind readonly

declare double @tan(double) nounwind readonly

define fastcc void @trace_line(i32 %line) nounwind {
entry:
	%.b3 = load i1* @current_surfaces.b		; <i1> [#uses=1]
	br i1 %.b3, label %bb, label %return

bb:		; preds = %bb9.i, %entry
	%.rle4 = phi double [ %8, %bb9.i ], [ 0.000000e+00, %entry ]		; <double> [#uses=1]
	%0 = load double* null, align 8		; <double> [#uses=3]
	%1 = fcmp une double %0, 0.000000e+00		; <i1> [#uses=1]
	br i1 %1, label %bb9.i, label %bb13.i

bb9.i:		; preds = %bb
	%2 = sub double %.rle4, %0		; <double> [#uses=0]
	%3 = tail call double @asin(double 0.000000e+00) nounwind readonly		; <double> [#uses=0]
	%4 = tail call double @sin(double 0.000000e+00) nounwind readonly		; <double> [#uses=1]
	%5 = mul double %4, %0		; <double> [#uses=1]
	%6 = tail call double @tan(double 0.000000e+00) nounwind readonly		; <double> [#uses=0]
	%7 = mul double %5, 0.000000e+00		; <double> [#uses=1]
	%8 = add double %7, 0.000000e+00		; <double> [#uses=1]
	br i1 false, label %return, label %bb

bb13.i:		; preds = %bb
	unreachable

return:		; preds = %bb9.i, %entry
	ret void
}
