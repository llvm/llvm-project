; RUN: llc < %s -relocation-model=dynamic-no-pic -mtriple=arm-apple-darwin9   | FileCheck %s

@x = external hidden global i32		; <i32*> [#uses=1]
@y = extern_weak hidden global i32	; <i32*> [#uses=1]

define i32 @t() nounwind readonly {
entry:
; CHECK: LCPI0_0:
; CHECK-NEXT: .long _x
;; .long _y can be used if @y is dso_local.
; CHECK: LCPI0_1:
; CHECK-NEXT: .long L_y$non_lazy_ptr

	%0 = load i32, i32* @x, align 4		; <i32> [#uses=1]
	%1 = load i32, i32* @y, align 4		; <i32> [#uses=1]
	%2 = add i32 %1, %0		; <i32> [#uses=1]
	ret i32 %2
}
