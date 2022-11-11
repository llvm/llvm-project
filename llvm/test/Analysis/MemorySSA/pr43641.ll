; RUN: opt -simple-loop-unswitch -enable-nontrivial-unswitch -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

; CHECK-LABEL: @c
define dso_local void @c(i32 signext %d) local_unnamed_addr {
entry:
  br i1 undef, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %tobool1 = icmp ne i32 %d, 0
  br label %while.body

while.body:                                       ; preds = %while.body, %while.body.lr.ph
  %call = tail call signext i32 @e()
  %0 = and i1 %tobool1, undef
  br i1 %0, label %while.body, label %while.end

while.end:                                        ; preds = %while.body, %entry
  ret void
}

declare signext i32 @e(...) local_unnamed_addr
