; RUN: llc < %s -mtriple=i386-apple-darwin9 | FileCheck %s

@x = common hidden global i32 0		; <ptr> [#uses=1]

define i32 @t() nounwind readonly {
entry:
; CHECK-LABEL: t:
; CHECK: movl _x, %eax
; CHECK: .comm _x,4
	%0 = load i32, ptr @x, align 4		; <i32> [#uses=1]
	ret i32 %0
}
