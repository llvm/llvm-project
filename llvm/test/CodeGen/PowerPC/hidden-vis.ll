; RUN: llc < %s -mtriple=powerpc-apple-darwin9 | not grep non_lazy_ptr

@x = weak hidden global i32 0		; <i32*> [#uses=1]

define i32 @t() nounwind readonly {
entry:
	%0 = load i32* @x, align 4		; <i32> [#uses=1]
	ret i32 %0
}
