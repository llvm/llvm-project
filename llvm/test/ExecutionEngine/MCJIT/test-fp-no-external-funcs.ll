; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

define double @test(double* %DP, double %Arg) {
	%D = load double* %DP		; <double> [#uses=1]
	%V = fadd double %D, 1.000000e+00		; <double> [#uses=2]
	%W = fsub double %V, %V		; <double> [#uses=3]
	%X = fmul double %W, %W		; <double> [#uses=2]
	%Y = fdiv double %X, %X		; <double> [#uses=2]
	%Q = fadd double %Y, %Arg		; <double> [#uses=1]
	%R = bitcast double %Q to double		; <double> [#uses=1]
	store double %Q, double* %DP
	ret double %Y
}

define i32 @main() {
	%X = alloca double		; <double*> [#uses=2]
	store double 0.000000e+00, double* %X
	call double @test( double* %X, double 2.000000e+00 )		; <double>:1 [#uses=0]
	ret i32 0
}

