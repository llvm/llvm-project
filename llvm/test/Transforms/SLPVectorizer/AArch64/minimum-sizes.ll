; RUN: opt -S -slp-vectorizer < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; This test ensures that we do not regress due to PR26364. The vectorizer
; should not compute a smaller size for %k.13 since it is in a use-def cycle
; and cannot be demoted.
;
; CHECK-LABEL: @PR26364
; CHECK: %k.13 = phi i32
;
define fastcc void @PR26364() {
entry:
  br i1 undef, label %for.end11, label %for.cond4

for.cond4:
  %k.13 = phi i32 [ undef, %entry ], [ %k.3, %for.cond4 ]
  %e.02 = phi i32 [ 1, %entry ], [ 0, %for.cond4 ]
  %e.1 = select i1 undef, i32 %e.02, i32 0
  %k.3 = select i1 undef, i32 %k.13, i32 undef
  br label %for.cond4

for.end11:
  ret void
}
