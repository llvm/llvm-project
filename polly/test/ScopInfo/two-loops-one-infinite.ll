; RUN: opt %loadNPMPolly '-passes=print<polly-detect>,print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; Verify we do not create a SCoP in the presence of infinite loops.
;
; CHECK-NOT:      Statements
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"

define void @foo(ptr noalias nocapture readonly %xxx, ptr noalias nocapture readonly %yyy, ptr nocapture readonly %zzz, i32 %conv6) {
while.body.us.preheader:
 %a2 = load ptr, ptr %zzz, align 4
 %sub = add nsw i32 %conv6, -1
  br label %while.body.us

while.body.us:                                    ; preds = %while.body.us.preheader, %if.then.us
  %uuu = phi i32 [ %www, %if.then.us ], [ 0, %while.body.us.preheader ]
  %a13 = load i32, ptr %yyy, align 8
  %vvv = icmp sgt i32 %a13, 0
  br i1 %vvv, label %while.body.13.us58.preheader, label %if.then.us

while.body.13.us58.preheader:                     ; preds = %while.body.us
  br label %while.body.13.us58

if.then.us:                                       ; preds = %while.body.us
  %add.us = add nuw nsw i32 %uuu, 1
  tail call void @goo(ptr %a2, i32 %uuu, ptr %a2, i32 %add.us, i32 %sub, i32 %a13) #3
  %www = add nuw nsw i32 %uuu, %conv6
  %a14 = load i32, ptr %xxx, align 4
  %cmp.us = icmp slt i32 %www, %a14
  br i1 %cmp.us, label %while.body.us, label %while.end.21.loopexit145

while.body.13.us58:
    br label %while.body.13.us58

while.end.21.loopexit145:
  ret void
}

declare void @goo(ptr, i32, ptr, i32, i32, i32) #1

