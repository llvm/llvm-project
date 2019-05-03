; RUN: llc -mtriple=x86_64-pc-linux %s -o - -regalloc=fast -optimize-regalloc=0 | FileCheck %s

; We used to consider the early clobber in the second asm statement as
; defining %0 before it was read. This caused us to omit the
; movq	-8(%rsp), %rdx

; CHECK: 	#APP
; CHECK-NEXT:	#NO_APP
; CHECK-NEXT:	movq	%rcx, %rax
; CHECK-NEXT:	movq	%rax, -8(%rsp)
; CHECK-NEXT:	movq	-8(%rsp), %rdx
; CHECK-NEXT:	#APP
; CHECK-NEXT:	#NO_APP
; CHECK-NEXT:	movq	%rdx, %rax
; CHECK-NEXT:	movq	%rdx, -8(%rsp)
; CHECK-NEXT:	ret

define i64 @foo() {
entry:
  %0 = tail call i64 asm "", "={cx}"() nounwind
  %1 = tail call i64 asm "", "=&r,0,r,~{rax}"(i64 %0, i64 %0) nounwind
  ret i64 %1
}
