; RUN: llc < %s -march=x86 -x86-asm-syntax=intel -mcpu=i486 | FileCheck %s
; Test that the load of the constant is folded into the operation.


define double @foo_add(double %P) {
	%tmp.1 = fadd double %P, 1.230000e+02		; <double> [#uses=1]
	ret double %tmp.1
}
; CHECK-LABEL: foo_add:
; CHECK: fadd dword ptr

define double @foo_mul(double %P) {
	%tmp.1 = fmul double %P, 1.230000e+02		; <double> [#uses=1]
	ret double %tmp.1
}
; CHECK-LABEL: foo_mul:
; CHECK: fmul dword ptr

define double @foo_sub(double %P) {
	%tmp.1 = fsub double %P, 1.230000e+02		; <double> [#uses=1]
	ret double %tmp.1
}
; CHECK-LABEL: foo_sub:
; CHECK: fadd dword ptr

define double @foo_subr(double %P) {
	%tmp.1 = fsub double 1.230000e+02, %P		; <double> [#uses=1]
	ret double %tmp.1
}
; CHECK-LABEL: foo_subr:
; CHECK: fsub qword ptr

define double @foo_div(double %P) {
	%tmp.1 = fdiv double %P, 1.230000e+02		; <double> [#uses=1]
	ret double %tmp.1
}
; CHECK-LABEL: foo_div:
; CHECK: fdiv dword ptr

define double @foo_divr(double %P) {
	%tmp.1 = fdiv double 1.230000e+02, %P		; <double> [#uses=1]
	ret double %tmp.1
}
; CHECK-LABEL: foo_divr:
; CHECK: fdiv qword ptr

