; RUN: opt %loadPolly \
; RUN: -polly-codegen -S < %s | FileCheck %s

; This test ensures that the expression N + 1 that is stored in the phi-node
; alloca, is directly computed and not incorrectly transfered through memory.

; CHECK: store i64 [[REG:%.*]], ptr %res.phiops
; CHECK: [[REG]] = add i64 %N, 1

define i64 @foo(ptr %A, i64 %N) {
entry:
  br label %next

next:
  %cond = icmp eq i64 %N, 0
  br i1 %cond, label %loop, label %merge

loop:
  %indvar = phi i64 [0, %next], [%indvar.next, %loop]
  %indvar.next = add i64 %indvar, 1
  %sum = add i64 %N, 1
  store float 4.0, ptr %A
  %cmp = icmp sle i64 %indvar.next, 100
  br i1 %cmp, label %loop, label %merge

merge:
  %res = phi i64 [%sum, %loop], [0, %next]
  br label %exit

exit:
  ret i64 %res
}
