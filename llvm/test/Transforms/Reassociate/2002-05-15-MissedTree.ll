; RUN: llvm-as < %s | opt -reassociate -instcombine -constprop -die | llvm-dis | not grep 5

define i32 @test(i32 %A, i32 %B) {
	%W = add i32 %B, -5		; <i32> [#uses=1]
	%Y = add i32 %A, 5		; <i32> [#uses=1]
	%Z = add i32 %W, %Y		; <i32> [#uses=1]
	ret i32 %Z
}

