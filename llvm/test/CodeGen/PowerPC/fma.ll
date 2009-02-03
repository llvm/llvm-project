; RUN: llvm-as < %s | llc -march=ppc32 | \
; RUN:   egrep {fn?madd|fn?msub} | count 8

define double @test_FMADD1(double %A, double %B, double %C) {
	%D = mul double %A, %B		; <double> [#uses=1]
	%E = add double %D, %C		; <double> [#uses=1]
	ret double %E
}

define double @test_FMADD2(double %A, double %B, double %C) {
	%D = mul double %A, %B		; <double> [#uses=1]
	%E = add double %D, %C		; <double> [#uses=1]
	ret double %E
}

define double @test_FMSUB(double %A, double %B, double %C) {
	%D = mul double %A, %B		; <double> [#uses=1]
	%E = sub double %D, %C		; <double> [#uses=1]
	ret double %E
}

define double @test_FNMADD1(double %A, double %B, double %C) {
	%D = mul double %A, %B		; <double> [#uses=1]
	%E = add double %D, %C		; <double> [#uses=1]
	%F = sub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
}

define double @test_FNMADD2(double %A, double %B, double %C) {
	%D = mul double %A, %B		; <double> [#uses=1]
	%E = add double %C, %D		; <double> [#uses=1]
	%F = sub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
}

define double @test_FNMSUB1(double %A, double %B, double %C) {
	%D = mul double %A, %B		; <double> [#uses=1]
	%E = sub double %C, %D		; <double> [#uses=1]
	ret double %E
}

define double @test_FNMSUB2(double %A, double %B, double %C) {
	%D = mul double %A, %B		; <double> [#uses=1]
	%E = sub double %D, %C		; <double> [#uses=1]
	%F = sub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
}

define float @test_FNMSUBS(float %A, float %B, float %C) {
	%D = mul float %A, %B		; <float> [#uses=1]
	%E = sub float %D, %C		; <float> [#uses=1]
	%F = sub float -0.000000e+00, %E		; <float> [#uses=1]
	ret float %F
}
