; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel -mcpu=i486 | \
; RUN:   grep {fadd\\|fsub\\|fdiv\\|fmul} | not grep -i ST

; Test that the load of the constant is folded into the operation.


define double @foo_add(double %P) {
	%tmp.1 = add double %P, 1.230000e+02		; <double> [#uses=1]
	ret double %tmp.1
}

define double @foo_mul(double %P) {
	%tmp.1 = mul double %P, 1.230000e+02		; <double> [#uses=1]
	ret double %tmp.1
}

define double @foo_sub(double %P) {
	%tmp.1 = sub double %P, 1.230000e+02		; <double> [#uses=1]
	ret double %tmp.1
}

define double @foo_subr(double %P) {
	%tmp.1 = sub double 1.230000e+02, %P		; <double> [#uses=1]
	ret double %tmp.1
}

define double @foo_div(double %P) {
	%tmp.1 = fdiv double %P, 1.230000e+02		; <double> [#uses=1]
	ret double %tmp.1
}

define double @foo_divr(double %P) {
	%tmp.1 = fdiv double 1.230000e+02, %P		; <double> [#uses=1]
	ret double %tmp.1
}
