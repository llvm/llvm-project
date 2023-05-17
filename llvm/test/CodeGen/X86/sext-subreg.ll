; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s
; rdar://7529457

define i64 @t(i64 %A, i64 %B, ptr %P, ptr%P2) nounwind {
; CHECK-LABEL: t:
; CHECK: movslq %e{{.*}}, %rax
; CHECK: movq %rax
; CHECK: movl %eax
  %C = add i64 %A, %B
  %D = trunc i64 %C to i32
  store volatile i32 %D, ptr %P
  %E = shl i64 %C, 32
  %F = ashr i64 %E, 32  
  store volatile i64 %F, ptr%P2
  store volatile i32 %D, ptr %P
  ret i64 undef
}
