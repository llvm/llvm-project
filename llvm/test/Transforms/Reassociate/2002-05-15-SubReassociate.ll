; With sub reassociation, constant folding can eliminate all of the constants.
;
; RUN: opt < %s -reassociate -constprop -instcombine -dce -S | not grep add

define i32 @test(i32 %A, i32 %B) {
	%W = add i32 5, %B		; <i32> [#uses=1]
	%X = add i32 -7, %A		; <i32> [#uses=1]
	%Y = sub i32 %X, %W		; <i32> [#uses=1]
	%Z = add i32 %Y, 12		; <i32> [#uses=1]
	ret i32 %Z
}

