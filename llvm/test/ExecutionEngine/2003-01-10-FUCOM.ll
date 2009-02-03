; RUN: llvm-as %s -f -o %t.bc
; RUN: lli %t.bc > /dev/null

define i32 @main() {
	%X = add double 0.000000e+00, 1.000000e+00		; <double> [#uses=1]
	%Y = sub double 0.000000e+00, 1.000000e+00		; <double> [#uses=2]
	%Z = fcmp oeq double %X, %Y		; <i1> [#uses=0]
	add double %Y, 0.000000e+00		; <double>:1 [#uses=0]
	ret i32 0
}

