; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep sub
; PR2553

define double @test(double %X) nounwind {
	; fsub of self can't be optimized away.
	%Y = sub double %X, %X
	ret double %Y
}
