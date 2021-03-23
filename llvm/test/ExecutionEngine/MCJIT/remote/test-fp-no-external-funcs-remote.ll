; RUN: %lli -remote-mcjit -mcjit-remote-process=lli-child-target%exeext %s > /dev/null
; XFAIL: windows-gnu,windows-msvc
; UNSUPPORTED: powerpc64-unknown-linux-gnu
; Remove UNSUPPORTED for powerpc64-unknown-linux-gnu if problem caused by r266663 is fixed

define double @test(double* %DP, double %Arg) nounwind {
	%D = load double, double* %DP		; <double> [#uses=1]
	%V = fadd double %D, 1.000000e+00		; <double> [#uses=2]
	%W = fsub double %V, %V		; <double> [#uses=3]
	%X = fmul double %W, %W		; <double> [#uses=2]
	%Y = fdiv double %X, %X		; <double> [#uses=2]
	%Q = fadd double %Y, %Arg		; <double> [#uses=1]
	%R = bitcast double %Q to double		; <double> [#uses=1]
	store double %Q, double* %DP
	ret double %Y
}

define i32 @main() nounwind {
	%X = alloca double		; <double*> [#uses=2]
	store double 0.000000e+00, double* %X
	call double @test( double* %X, double 2.000000e+00 )		; <double>:1 [#uses=0]
	ret i32 0
}
