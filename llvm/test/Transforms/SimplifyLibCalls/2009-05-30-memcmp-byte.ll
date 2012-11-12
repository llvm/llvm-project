; RUN: opt < %s -simplify-libcalls -instcombine -S | grep "ret i32 -65"
; PR4284

define i32 @test() nounwind {
entry:
	%c0 = alloca i8, align 1		; <i8*> [#uses=2]
	%c2 = alloca i8, align 1		; <i8*> [#uses=2]
	store i8 64, i8* %c0
	store i8 -127, i8* %c2
	%call = call i32 @memcmp(i8* %c0, i8* %c2, i32 1)		; <i32> [#uses=1]
	ret i32 %call
}

declare i32 @memcmp(i8*, i8*, i32)
