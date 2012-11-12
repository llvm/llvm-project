; RUN: llc < %s -march=ppc32 -fp-contract=fast | \
; RUN:   egrep "fn?madd|fn?msub" | count 8

define double @test_FMADD1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	ret double %E
}

define double @test_FMADD2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	ret double %E
}

define double @test_FMSUB(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %D, %C		; <double> [#uses=1]
	ret double %E
}

define double @test_FNMADD1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
}

define double @test_FNMADD2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %C, %D		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
}

define double @test_FNMSUB1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %C, %D		; <double> [#uses=1]
	ret double %E
}

define double @test_FNMSUB2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %D, %C		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
}

define float @test_FNMSUBS(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fsub float %D, %C		; <float> [#uses=1]
	%F = fsub float -0.000000e+00, %E		; <float> [#uses=1]
	ret float %F
}
