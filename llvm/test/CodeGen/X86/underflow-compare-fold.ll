; RUN: llc < %s -mtriple=x86_64 | FileCheck %s

; GitHub issue #161036

define i64 @subIfNoUnderflow_umin(i64 %a, i64 %b) {
; CHECK-LABEL: subIfNoUnderflow_umin
; CHECK-LABEL: %bb.0
; CHECK-NEXT: movq    %rdi, %rax
; CHECK-NEXT: subq    %rsi, %rax
; CHECK-NEXT: cmovbq  %rdi, %rax
; retq
entry:
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %sub, i64 %a)
  ret i64 %cond
}
