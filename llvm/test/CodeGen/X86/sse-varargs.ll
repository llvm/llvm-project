; RUN: llc < %s -march=x86 -mattr=+sse2 | grep xmm | grep esp

define i32 @t() nounwind  {
entry:
	tail call void (i32, ...) @foo( i32 1, <4 x i32> < i32 10, i32 11, i32 12, i32 13 > ) nounwind 
	ret i32 0
}

declare void @foo(i32, ...)
