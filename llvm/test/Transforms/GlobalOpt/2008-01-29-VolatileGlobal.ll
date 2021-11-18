; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK: load volatile
@t0.1441 = internal global double 0x3FD5555555555555, align 8		; <double*> [#uses=1]

define double @foo() nounwind  {
entry:
	%tmp1 = load volatile double, double* @t0.1441, align 8		; <double> [#uses=2]
	%tmp4 = fmul double %tmp1, %tmp1		; <double> [#uses=1]
	ret double %tmp4
}
