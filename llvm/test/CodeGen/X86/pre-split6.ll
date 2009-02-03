; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -pre-alloc-split

@current_surfaces.b = external global i1		; <i1*> [#uses=1]

declare double @sin(double) nounwind readonly

declare double @asin(double) nounwind readonly

define fastcc void @trace_line(i32 %line) nounwind {
entry:
	%.b3 = load i1* @current_surfaces.b		; <i1> [#uses=1]
	br i1 %.b3, label %bb.nph, label %return

bb.nph:		; preds = %entry
	%0 = load double* null, align 8		; <double> [#uses=1]
	%1 = load double* null, align 8		; <double> [#uses=2]
	%2 = fcmp une double %0, 0.000000e+00		; <i1> [#uses=1]
	br i1 %2, label %bb9.i, label %bb13.i

bb9.i:		; preds = %bb.nph
	%3 = tail call double @asin(double 0.000000e+00) nounwind readonly		; <double> [#uses=0]
	%4 = fdiv double 1.000000e+00, %1		; <double> [#uses=1]
	%5 = mul double %4, 0.000000e+00		; <double> [#uses=1]
	%6 = tail call double @asin(double %5) nounwind readonly		; <double> [#uses=0]
	unreachable

bb13.i:		; preds = %bb.nph
	%7 = fdiv double 1.000000e+00, %1		; <double> [#uses=1]
	%8 = tail call double @sin(double 0.000000e+00) nounwind readonly		; <double> [#uses=1]
	%9 = mul double %7, %8		; <double> [#uses=1]
	%10 = tail call double @asin(double %9) nounwind readonly		; <double> [#uses=0]
	unreachable

return:		; preds = %entry
	ret void
}
