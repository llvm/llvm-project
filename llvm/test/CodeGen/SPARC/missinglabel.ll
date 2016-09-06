; RUN: llc < %s -verify-machineinstrs | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64-S128"
target triple = "sparc64-unknown-linux-gnu"

define void @f() align 2 {
entry:
; CHECK: %xcc, .LBB0_2
  %cmp = icmp eq i64 undef, 0
  br i1 %cmp, label %targetblock, label %cond.false

cond.false:
  unreachable

; CHECK: .LBB0_2: ! %targetblock
targetblock:
  br i1 undef, label %cond.false.i83, label %exit.i85

cond.false.i83:
  unreachable

exit.i85:
  unreachable
}
