; Inlining used to break PHI nodes.  This tests that they are correctly updated
; when a node is split around the call instruction.  The verifier caught the error.
;
; RUN: llvm-as < %s | opt -inline
;

define i64 @test(i64 %X) {
	ret i64 %X
}

define i64 @fib(i64 %n) {
; <label>:0
	%T = icmp ult i64 %n, 2		; <i1> [#uses=1]
	br i1 %T, label %BaseCase, label %RecurseCase

RecurseCase:		; preds = %0
	%result = call i64 @test( i64 %n )		; <i64> [#uses=0]
	br label %BaseCase

BaseCase:		; preds = %RecurseCase, %0
	%X = phi i64 [ 1, %0 ], [ 2, %RecurseCase ]		; <i64> [#uses=1]
	ret i64 %X
}
