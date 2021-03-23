; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null


define i32 @main() {
	%double1 = fadd double 0.000000e+00, 0.000000e+00		; <double> [#uses=6]
	%double2 = fadd double 0.000000e+00, 0.000000e+00		; <double> [#uses=6]
	%float1 = fadd float 0.000000e+00, 0.000000e+00		; <float> [#uses=6]
	%float2 = fadd float 0.000000e+00, 0.000000e+00		; <float> [#uses=6]
	%test49 = fcmp oeq float %float1, %float2		; <i1> [#uses=0]
	%test50 = fcmp oge float %float1, %float2		; <i1> [#uses=0]
	%test51 = fcmp ogt float %float1, %float2		; <i1> [#uses=0]
	%test52 = fcmp ole float %float1, %float2		; <i1> [#uses=0]
	%test53 = fcmp olt float %float1, %float2		; <i1> [#uses=0]
	%test54 = fcmp une float %float1, %float2		; <i1> [#uses=0]
	%test55 = fcmp oeq double %double1, %double2		; <i1> [#uses=0]
	%test56 = fcmp oge double %double1, %double2		; <i1> [#uses=0]
	%test57 = fcmp ogt double %double1, %double2		; <i1> [#uses=0]
	%test58 = fcmp ole double %double1, %double2		; <i1> [#uses=0]
	%test59 = fcmp olt double %double1, %double2		; <i1> [#uses=0]
	%test60 = fcmp une double %double1, %double2		; <i1> [#uses=0]
	ret i32 0
}


