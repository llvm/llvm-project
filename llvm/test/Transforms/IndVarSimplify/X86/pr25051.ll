; RUN: opt -S -passes=indvars < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @somefunc(ptr %arr) {
; CHECK-LABEL: @somefunc(
entry:
  br label %for.cond.1.preheader

for.cond.1.preheader:                             ; preds = %for.inc.9, %entry
  %index3.013 = phi i32 [ 0, %entry ], [ %index3.1.lcssa, %for.inc.9 ]
  %index.012 = phi i32 [ 0, %entry ], [ %inc10, %for.inc.9 ]
  %cmp2.9 = icmp sgt i32 %index.012, 0
  br i1 %cmp2.9, label %for.body.3.lr.ph, label %for.inc.9

for.body.3.lr.ph:                                 ; preds = %for.cond.1.preheader
  %idxprom5 = sext i32 %index.012 to i64 
  br label %for.body.3

for.body.3:                                       ; preds = %for.body.3, %for.body.3.lr.ph
  %index3.111 = phi i32 [ %index3.013, %for.body.3.lr.ph ], [ %inc, %for.body.3 ]
  %index2.010 = phi i32 [ 0, %for.body.3.lr.ph ], [ %inc8, %for.body.3 ]
  %inc = add nsw i32 %index3.111, 1
  %idxprom = sext i32 %index3.111 to i64
  %arrayidx = getelementptr inbounds double, ptr %arr, i64 %idxprom
  %idxprom4 = sext i32 %index2.010 to i64
  %inc8 = add nsw i32 %index2.010, 1
  %cmp2 = icmp slt i32 %inc8, %index.012
  br i1 %cmp2, label %for.body.3, label %for.inc.9.loopexit

for.inc.9.loopexit:                               ; preds = %for.body.3
  %inc.lcssa = phi i32 [ %inc, %for.body.3 ]
  br label %for.inc.9

for.inc.9:                                        ; preds = %for.inc.9.loopexit, %for.cond.1.preheader
  %index3.1.lcssa = phi i32 [ %index3.013, %for.cond.1.preheader ], [ %inc.lcssa, %for.inc.9.loopexit ]
  %inc10 = add nsw i32 %index.012, 1
  %cmp = icmp slt i32 %inc10, 10
  br i1 %cmp, label %for.cond.1.preheader, label %for.end.11

for.end.11:                                       ; preds = %for.inc.9
  ret i32 1
}
