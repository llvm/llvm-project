; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep -i ST | not grep {fadd\\|fsub\\|fdiv\\|fmul}

; Test that the load of the memory location is folded into the operation.

define double @test_add(double %X, double* %P) {
	%Y = load double* %P		; <double> [#uses=1]
	%R = add double %X, %Y		; <double> [#uses=1]
	ret double %R
}

define double @test_mul(double %X, double* %P) {
	%Y = load double* %P		; <double> [#uses=1]
	%R = mul double %X, %Y		; <double> [#uses=1]
	ret double %R
}

define double @test_sub(double %X, double* %P) {
	%Y = load double* %P		; <double> [#uses=1]
	%R = sub double %X, %Y		; <double> [#uses=1]
	ret double %R
}

define double @test_subr(double %X, double* %P) {
	%Y = load double* %P		; <double> [#uses=1]
	%R = sub double %Y, %X		; <double> [#uses=1]
	ret double %R
}

define double @test_div(double %X, double* %P) {
	%Y = load double* %P		; <double> [#uses=1]
	%R = fdiv double %X, %Y		; <double> [#uses=1]
	ret double %R
}

define double @test_divr(double %X, double* %P) {
	%Y = load double* %P		; <double> [#uses=1]
	%R = fdiv double %Y, %X		; <double> [#uses=1]
	ret double %R
}
