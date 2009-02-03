; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9   | grep mov | count 1
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin9 | not grep GOT

@x = weak hidden global i32 0		; <i32*> [#uses=1]

define i32 @t() nounwind readonly {
entry:
	%0 = load i32* @x, align 4		; <i32> [#uses=1]
	ret i32 %0
}
