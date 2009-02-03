; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep shl
; END.

define i1 @test0(i39 %X, i39 %A) {
	%B = lshr i39 %X, %A
	%D = trunc i39 %B to i1
	ret i1 %D
}

define i1 @test1(i799 %X, i799 %A) {
	%B = lshr i799 %X, %A
	%D = trunc i799 %B to i1
	ret i1 %D
}
