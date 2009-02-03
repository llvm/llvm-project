; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -pre-alloc-split

@object_distance = external global double, align 8		; <double*> [#uses=1]
@axis_slope_angle = external global double, align 8		; <double*> [#uses=1]
@current_surfaces.b = external global i1		; <i1*> [#uses=1]

declare double @sin(double) nounwind readonly

declare double @asin(double) nounwind readonly

declare double @tan(double) nounwind readonly

define fastcc void @trace_line(i32 %line) nounwind {
entry:
	%.b3 = load i1* @current_surfaces.b		; <i1> [#uses=1]
	br i1 %.b3, label %bb, label %return

bb:		; preds = %bb, %entry
	%0 = tail call double @asin(double 0.000000e+00) nounwind readonly		; <double> [#uses=1]
	%1 = add double 0.000000e+00, %0		; <double> [#uses=2]
	%2 = tail call double @asin(double 0.000000e+00) nounwind readonly		; <double> [#uses=1]
	%3 = sub double %1, %2		; <double> [#uses=2]
	store double %3, double* @axis_slope_angle, align 8
	%4 = fdiv double %1, 2.000000e+00		; <double> [#uses=1]
	%5 = tail call double @sin(double %4) nounwind readonly		; <double> [#uses=1]
	%6 = mul double 0.000000e+00, %5		; <double> [#uses=1]
	%7 = tail call double @tan(double %3) nounwind readonly		; <double> [#uses=0]
	%8 = add double 0.000000e+00, %6		; <double> [#uses=1]
	store double %8, double* @object_distance, align 8
	br label %bb

return:		; preds = %entry
	ret void
}
