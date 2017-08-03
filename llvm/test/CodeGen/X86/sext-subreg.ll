; RUN: llc < %s -march=x86-64 | FileCheck %s
; rdar://7529457

define i64 @t(i64 %A, i64 %B, i32* %P, i64 *%P2) nounwind {
; CHECK-LABEL: t:
; CHECK: movslq %e{{.*}}, %rax
; CHECK: movq %rax
; CHECK: movl %eax
  %C = add i64 %A, %B
  %D = trunc i64 %C to i32
  store volatile i32 %D, i32* %P
  %E = shl i64 %C, 32
  %F = ashr i64 %E, 32  
  store volatile i64 %F, i64 *%P2
  store volatile i32 %D, i32* %P
  ret i64 undef
}
