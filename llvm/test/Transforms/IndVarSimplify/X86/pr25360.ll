; RUN: opt -passes=indvars -S < %s | FileCheck %s


; Ensure that does not crash

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f() {
; CHECK-LABEL: @f(
entry:
  br label %for.end

for.condt:                         ; preds = %for.end
  br i1 true, label %for.cond.0, label %for.end

for.end:                                          ; preds = %for.body.3
  %inc = select i1 undef, i32 2, i32 1
  br i1 false, label %for.condt, label %for.cond.0

for.cond.0:                       ; preds = %for.end, %for.condt
  %init = phi i32 [ 0, %for.condt ], [ %inc, %for.end ]
  br i1 true, label %for.end.13, label %for.body.9

for.body.9:                                       ; preds = %for.body.9, %for.cond.0
  %p1.addr.22 = phi i32 [ %inc10, %for.body.9 ], [ %init, %for.cond.0 ]
  %inc10 = add i32 %p1.addr.22, 1
  br i1 true, label %for.end.13, label %for.body.9

for.end.13:                                       ; preds = %for.cond.7.for.end.13_crit_edge, %for.cond.0
  %p1.addr.2.lcssa = phi i32 [ %inc10, %for.body.9 ], [ %init, %for.cond.0 ]
  ret void
}
