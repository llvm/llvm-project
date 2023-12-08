; RUN: opt < %s -passes=instcombine -S | grep sub
; PR2553

define double @test(double %X) nounwind {
	; fsub of self can't be optimized away.
	%Y = fsub double %X, %X
	ret double %Y
}
