; Testcase for calls to the standard C "pow" function
;
; RUN: opt < %s -simplify-libcalls -S | not grep "call .pow"


declare double @pow(double, double)
declare float @powf(float, float)

define double @test1(double %X) {
	%Y = call double @pow( double %X, double 0.000000e+00 )		; <double> [#uses=1]
	ret double %Y
}

define double @test2(double %X) {
	%Y = call double @pow( double %X, double -0.000000e+00 )		; <double> [#uses=1]
	ret double %Y
}

define double @test3(double %X) {
	%Y = call double @pow( double 1.000000e+00, double %X )		; <double> [#uses=1]
	ret double %Y
}

define double @test4(double %X) {
	%Y = call double @pow( double %X, double 2.0)
	ret double %Y
}

define float @test4f(float %X) {
	%Y = call float @powf( float %X, float 2.0)
	ret float %Y
}

define float @test5f(float %X) {
	%Y = call float @powf(float 2.0, float %X)  ;; exp2
	ret float %Y
}
