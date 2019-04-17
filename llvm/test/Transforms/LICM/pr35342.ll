; RUN: opt -licm -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @f1
; CHECK-LABEL: bci_524:
; CHECK: add i32 undef, undef
define void @f1(i32 %v) {
not_zero.lr.ph:
  br label %not_zero

not_zero:
  br i1 undef, label %bci_748 ,  label %bci_314

bci_314:
  %0 = select i1 undef, i32 undef, i32 undef
  br label %not_zero

bci_524:                   ; No predecessors!
  %add = add i32 %0, %0
  br label %bci_748

bci_748:
  ret void
}
