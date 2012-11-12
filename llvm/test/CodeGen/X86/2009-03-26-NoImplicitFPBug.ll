; RUN: llc < %s -march=x86 -mattr=+sse2

define double @t(double %x) nounwind ssp noimplicitfloat {
entry:
	br i1 false, label %return, label %bb3

bb3:		; preds = %entry
	ret double 0.000000e+00

return:		; preds = %entry
	ret double undef
}
