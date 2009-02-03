; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep {volatile load}
@t0.1441 = internal global double 0x3FD5555555555555, align 8		; <double*> [#uses=1]

define double @foo() nounwind  {
entry:
	%tmp1 = volatile load double* @t0.1441, align 8		; <double> [#uses=2]
	%tmp4 = mul double %tmp1, %tmp1		; <double> [#uses=1]
	ret double %tmp4
}
