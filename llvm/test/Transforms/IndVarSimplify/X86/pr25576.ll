; RUN: opt -S -passes=indvars < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @fn1(i1 %arg) {
; CHECK-LABEL: @fn1(
entry:
  br label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.inc7, %for.cond.loopexit, %entry
  %c.1.lcssa = phi i32 [ %inc8, %for.inc7 ], [ 0, %for.cond.loopexit ], [ 0, %entry ]
  br i1 %arg, label %for.cond.loopexit, label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc7, %for.cond.loopexit
  %c.17 = phi i32 [ %inc8, %for.inc7 ], [ 0, %for.cond.loopexit ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %inc14 = phi i32 [ 0, %for.cond4.preheader ], [ %inc, %for.body6 ]
  %idxprom = zext i32 %inc14 to i64
  %inc = add i32 %inc14, 1
  %cmp5 = icmp ult i32 %inc, 2
  br i1 %cmp5, label %for.body6, label %for.inc7

for.inc7:                                         ; preds = %for.body6
  %inc.lcssa = phi i32 [ %inc, %for.body6 ]
  %inc8 = add i32 %c.17, 1
  %cmp = icmp ult i32 %inc8, %inc.lcssa
  br i1 %cmp, label %for.cond4.preheader, label %for.cond.loopexit
}
