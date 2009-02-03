; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | grep ldr | count 6
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | grep non_lazy_ptr
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | grep long | count 4

@x = external hidden global i32		; <i32*> [#uses=1]
@y = extern_weak hidden global i32	; <i32*> [#uses=1]

define i32 @t() nounwind readonly {
entry:
	%0 = load i32* @x, align 4		; <i32> [#uses=1]
	%1 = load i32* @y, align 4		; <i32> [#uses=1]
	%2 = add i32 %1, %0		; <i32> [#uses=1]
	ret i32 %2
}
