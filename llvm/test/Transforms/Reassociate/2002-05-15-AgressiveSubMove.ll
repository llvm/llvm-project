; RUN: opt < %s -reassociate -instcombine -constprop -dce -S | not grep add

define i32 @test(i32 %A) {
	%X = add i32 %A, 1		; <i32> [#uses=1]
	%Y = add i32 %A, 1		; <i32> [#uses=1]
	%r = sub i32 %X, %Y		; <i32> [#uses=1]
	ret i32 %r
}

