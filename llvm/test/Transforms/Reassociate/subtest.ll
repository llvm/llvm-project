; With sub reassociation, constant folding can eliminate the 12 and -12 constants.
;
; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | not grep 12

define i32 @test(i32 %A, i32 %B) {
	%X = add i32 -12, %A		; <i32> [#uses=1]
	%Y = sub i32 %X, %B		; <i32> [#uses=1]
	%Z = add i32 %Y, 12		; <i32> [#uses=1]
	ret i32 %Z
}

