; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; PR3324
define double @f1(double %a, double %b, double %c, double %d, double %e, double %f, double %g) nounwind {
entry:
	br i1 false, label %bb113, label %bb129

bb113:		; preds = %entry
	ret double 0.000000e+00

bb129:		; preds = %entry
	%tmp134 = sub double %b, %a		; <double> [#uses=1]
	%tmp136 = sub double %tmp134, %c		; <double> [#uses=1]
	%tmp138 = add double %tmp136, %d		; <double> [#uses=1]
	%tmp140 = sub double %tmp138, %e		; <double> [#uses=1]
	%tmp142 = add double %tmp140, %f		; <double> [#uses=1]
	%tmp.0 = mul double %tmp142, 0.000000e+00		; <double> [#uses=1]
	ret double %tmp.0
}
