; RUN: llc -asm-verbose=false -mtriple=x86_64-apple-darwin -o - < %s | FileCheck %s

; LSR should leave non-affine expressions alone because it currently
; doesn't know how to do anything with them, and when it tries, it
; gets SCEVExpander's current expansion for them, which is suboptimal.

; CHECK:        xorl %eax, %eax
; CHECK-NEXT:   align
; CHECK-NEXT: BB0_1:
; CHECK-NEXT:   movq  %rax, (%rdx)
; CHECK-NEXT:   addq  %rsi, %rax
; CHECK-NEXT:   cmpq  %rdi, %rax
; CHECK-NEXT:   jl
; CHECK-NEXT:   imulq %rax, %rax
; CHECK-NEXT:   ret
define i64 @foo(i64 %n, i64 %s, i64* %p) nounwind {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  store volatile i64 %i, i64* %p
  %i.next = add i64 %i, %s
  %c = icmp slt i64 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  %mul = mul i64 %i.next, %i.next
  ret i64 %mul
}
