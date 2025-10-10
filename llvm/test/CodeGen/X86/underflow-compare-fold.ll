; RUN: llc < %s -mtriple=x86_64 | FileCheck %s

; GitHub issue #161036

define i64 @underflow_compare_fold(i64 %a, i64 %b) {
; CHECK-LABEL: underflow_compare_fold
; CHECK-LABEL: %bb.0
; CHECK-NEXT: movq    %rdi, %rax
; CHECK-NEXT: subq    %rsi, %rax
; CHECK-NEXT: cmovbq  %rdi, %rax
; CHECK-NEXT: retq
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %sub, i64 %a)
  ret i64 %cond
}
