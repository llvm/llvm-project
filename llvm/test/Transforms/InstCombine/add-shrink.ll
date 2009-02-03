; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {add i32}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep sext | count 1

; Should only have one sext and the add should be i32 instead of i64.

define i64 @test1(i32 %A) {
	%B = ashr i32 %A, 7		; <i32> [#uses=1]
	%C = ashr i32 %A, 9		; <i32> [#uses=1]
	%D = sext i32 %B to i64		; <i64> [#uses=1]
	%E = sext i32 %C to i64		; <i64> [#uses=1]
	%F = add i64 %D, %E		; <i64> [#uses=1]
	ret i64 %F
}

