; RUN: opt -S -passes=indvars < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; SCEVExpander would try to RAUW %val_2 with %c.lcssa, breaking "def
; dominates uses".

define void @pr27232(i32 %val) {
; CHECK-LABEL: @pr27232(
entry:
  br i1 undef, label %loop_0.cond, label %for.body.us

for.body.us:
  br label %loop_0.cond

loop_0.cond:
  %val_2 = phi i32 [ %val, %for.body.us ], [ undef, %entry ]
  br i1 true, label %loop_0.ph, label %loop_1.ph

loop_0.ph:
  br label %loop_0

loop_1.exit:
  br label %loop_1.ph

loop_1.ph:
  %c.lcssa = phi i32 [ 0, %loop_0.cond ], [ %val_2, %loop_1.exit ]
  br label %loop_1

loop_0:
  br i1 undef, label %loop_0, label %loop_1.exit

loop_1:
  %d.1 = phi i32 [ %c.lcssa, %loop_1 ], [ %val_2, %loop_1.ph ]
  %t.1 = phi i32 [ %val_2, %loop_1 ], [ %c.lcssa, %loop_1.ph ]
  br i1 undef, label %leave, label %loop_1

leave:
  ret void
}


; @ReplaceArg_0 and @ReplaceArg_1 used to trigger a failed cast<>
; assertion in SCEVExpander.

define void @ReplaceArg_0(i32 %val) {
; CHECK-LABEL: @ReplaceArg_0(
entry:
  br i1 undef, label %loop_0.cond, label %for.body.us

for.body.us:
  br label %loop_0.cond

loop_0.cond:
  br i1 true, label %loop_0.ph, label %loop_1.ph

loop_0.ph:
  br label %loop_0

loop_1.exit:
  br label %loop_1.ph

loop_1.ph:
  %c.lcssa = phi i32 [ 0, %loop_0.cond ], [ %val, %loop_1.exit ]
  br label %loop_1

loop_0:
  br i1 undef, label %loop_0, label %loop_1.exit

loop_1:
  %d.1 = phi i32 [ %c.lcssa, %loop_1 ], [ %val, %loop_1.ph ]
  %t.1 = phi i32 [ %val, %loop_1 ], [ %c.lcssa, %loop_1.ph ]
  br i1 undef, label %leave, label %loop_1

leave:
  ret void
}

define void @ReplaceArg_1(i32 %val) {
; CHECK-LABEL: @ReplaceArg_1(
entry:
  br i1 undef, label %loop_0.cond, label %for.body.us

for.body.us:
  br label %loop_0.cond

loop_0.cond:
  br i1 true, label %loop_0.ph, label %loop_1.ph

loop_0.ph:
  br label %loop_0

loop_1.exit:
  br label %loop_1.ph

loop_1.ph:
  %c.lcssa = phi i32 [ 0, %loop_0.cond ], [ %val, %loop_1.exit ]
  br label %loop_1

loop_0:
  br i1 undef, label %loop_0, label %loop_1.exit

loop_1:
  %t.1 = phi i32 [ %val, %loop_1 ], [ %c.lcssa, %loop_1.ph ]
  %d.1 = phi i32 [ %c.lcssa, %loop_1 ], [ %val, %loop_1.ph ]
  br i1 undef, label %leave, label %loop_1

leave:
  ret void
}
