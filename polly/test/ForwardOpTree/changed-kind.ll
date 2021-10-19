; RUN: opt %loadPolly -polly-optree -analyze < %s | FileCheck %s -match-full-lines

; In the code below, %0 is known to be equal to the content of @c (constant 0).
; Thus, in order to save a scalar dependency, forward-optree replaces
; the use of %0 in Stmt_lor_end93 by a load from @c by changing the
; access find from a scalar access to a array accesses.
; llvm.org/PR48034 decribes a crash caused by the mid-processing change.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c = external dso_local global i64, align 8

define void @func()  {
entry:
  br label %lor.end

while.cond.loopexit:
  %conv102.le = trunc i64 %xor101 to i8
  ret void

lor.end:
  %tobool72.not = icmp eq i64 0, 0
  br i1 %tobool72.not, label %lor.rhs87, label %lor.end.thread

lor.end.thread:
  br label %lor.rhs87

lor.rhs87:
  %0 = phi i64 [ 0, %lor.end.thread ], [ 0, %lor.end ]
  store i64 %0, i64* @c, align 8
  %neg79 = xor i64 %0, -1
  br label %lor.end93

lor.end93:
  %tobool93 = icmp ne i64 undef, 0
  %conv95 = zext i1 %tobool93 to i64
  %and100 = and i64 %conv95, undef
  %xor101 = xor i64 %and100, %neg79
  %xor103 = xor i64 %0, %conv95
  br label %while.cond.loopexit
}


; CHECK: Statistics {
; CHECK:     Reloads: 1
; CHECK: }

; CHECK: After statements {
; CHECK:     Stmt_lor_end93
; CHECK-NEXT:        ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:            { Stmt_lor_end93[] -> MemRef3[] };
; CHECK-NEXT:       new: { Stmt_lor_end93[] -> MemRef_c[0] };
; CHECK: }
