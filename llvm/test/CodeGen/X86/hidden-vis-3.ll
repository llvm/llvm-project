; RUN: llc < %s -mtriple=i386-apple-darwin9   | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-apple-darwin9 | FileCheck %s -check-prefix=X64

@x = external hidden global i32		; <i32*> [#uses=1]
@y = extern_weak hidden global i32	; <i32*> [#uses=1]

define i32 @t() nounwind readonly {
entry:
;; movl _y, %eax can be used if @y is dso_local.
; X32: _t:
; X32:      movl L_y$non_lazy_ptr, %eax
; X32-NEXT: movl (%eax), %eax

; X64: _t:
; X64: movq _y@GOTPCREL(%rip), %rax

	%0 = load i32, i32* @x, align 4		; <i32> [#uses=1]
	%1 = load i32, i32* @y, align 4		; <i32> [#uses=1]
	%2 = add i32 %1, %0		; <i32> [#uses=1]
	ret i32 %2
}
