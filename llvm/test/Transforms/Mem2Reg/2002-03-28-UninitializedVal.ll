; Uninitialized values are not handled correctly.
;
; RUN: opt < %s -mem2reg -disable-output
;

define i32 @test() {
        ; To be promoted
	%X = alloca i32		; <i32*> [#uses=1]
	%Y = load i32, i32* %X		; <i32> [#uses=1]
	ret i32 %Y
}
