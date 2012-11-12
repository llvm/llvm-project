; RUN: llc < %s -relocation-model=dynamic-no-pic -mtriple=arm-apple-darwin | FileCheck %s

@x = weak hidden global i32 0		; <i32*> [#uses=1]

define i32 @t() nounwind readonly {
entry:
; CHECK: t:
; CHECK: ldr
; CHECK-NEXT: ldr
	%0 = load i32* @x, align 4		; <i32> [#uses=1]
	ret i32 %0
}
