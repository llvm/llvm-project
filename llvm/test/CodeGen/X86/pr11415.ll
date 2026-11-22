; RUN: llc -mtriple=x86_64-pc-linux %s -o - -regalloc=fast | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux -O0 -regalloc-fast-tied %s -o - | FileCheck --check-prefix=O0 %s

; We used to consider the early clobber in the second asm statement as
; defining %0 before it was read. This caused us to omit the
; movq	-8(%rsp), %rdx

; CHECK: 	#APP
; CHECK-NEXT:	#NO_APP
; CHECK-NEXT:	movq	%rcx, %rdx
; CHECK-NEXT:	#APP
; CHECK-NEXT:	#NO_APP
; CHECK-NEXT:	movq	%rcx, -8(%rsp)
; CHECK-NEXT:	movq	-8(%rsp), %rax
; CHECK-NEXT:	ret

;; Each asm operand reads its own copy of %0, so the tied use takes over the
;; early-clobber def's %rcx while the other operand stays in %rdx.
; O0: 	#APP
; O0-NEXT:	#NO_APP
; O0-NEXT:	movq	%rcx, %rdx
; O0-NEXT:	movq	%rdx, %rcx
; O0-NEXT:	#APP
; O0-NEXT:	#NO_APP
; O0-NEXT:	movq	%rcx, -8(%rsp)
; O0-NEXT:	movq	-8(%rsp), %rax
; O0-NEXT:	retq

define i64 @foo() {
entry:
  %0 = tail call i64 asm "", "={cx}"() nounwind
  %1 = tail call i64 asm "", "=&r,0,r,~{rax}"(i64 %0, i64 %0) nounwind
  ret i64 %1
}
