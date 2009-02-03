; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9 | grep non_lazy_ptr
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9 | grep long
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9 | grep comm

@x = common hidden global i32 0		; <i32*> [#uses=1]

define i32 @t() nounwind readonly {
entry:
	%0 = load i32* @x, align 4		; <i32> [#uses=1]
	ret i32 %0
}
