; RUN: %lli %s > /dev/null

define i32 @main() {
	%X = fadd double 0.000000e+00, 1.000000e+00		; <double> [#uses=1]
	%Y = fsub double 0.000000e+00, 1.000000e+00		; <double> [#uses=2]
	%Z = fcmp oeq double %X, %Y		; <i1> [#uses=0]
	fadd double %Y, 0.000000e+00		; <double>:1 [#uses=0]
	ret i32 0
}

