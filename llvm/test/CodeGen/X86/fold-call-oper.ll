; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
;
; PR18396: Assertion: MO->isDead "Cannot fold physreg def".
; InlineSpiller::foldMemoryOperand needs to handle undef call operands.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@a = external global i32**, align 8
@b = external global i32, align 4

; Check that the call targets are folded, and we don't crash!
; CHECK-LABEL: foldCallOper:
; CHECK: callq *{{.*}}(%rbp)
; CHECK: callq *{{.*}}(%rbp)
define void @foldCallOper(i32 (i32*, i32, i32**)* nocapture %p1) #0 {
entry:
  %0 = load i32**, i32*** @a, align 8
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %exitcond5.i = icmp eq i32 undef, undef
  br i1 %exitcond5.i, label %for.body3.lr.ph.i, label %for.body.i

for.body3.lr.ph.i:                                ; preds = %for.body.i
  %call.i = tail call i32 %p1(i32* undef, i32 0, i32** null)
  %tobool.i = icmp eq i32 %call.i, 0
  br label %for.body3.i

for.body3.i:                                      ; preds = %for.inc8.i, %for.body3.lr.ph.i
  %1 = phi i32* [ undef, %for.body3.lr.ph.i ], [ %.pre.i, %for.inc8.i ]
  %indvars.iv.i = phi i64 [ 1, %for.body3.lr.ph.i ], [ %phitmp.i, %for.inc8.i ]
  %call5.i = tail call i32 %p1(i32* %1, i32 0, i32** %0)
  br i1 %tobool.i, label %for.inc8.i, label %if.then.i

if.then.i:                                        ; preds = %for.body3.i
  %2 = load i32, i32* %1, align 4
  store i32 %2, i32* @b, align 4
  br label %for.inc8.i

for.inc8.i:                                       ; preds = %if.then.i, %for.body3.i
  %lftr.wideiv.i = trunc i64 %indvars.iv.i to i32
  %arrayidx4.phi.trans.insert.i = getelementptr inbounds [0 x i32*], [0 x i32*]* undef, i64 0, i64 %indvars.iv.i
  %.pre.i = load i32*, i32** %arrayidx4.phi.trans.insert.i, align 8
  %phitmp.i = add i64 %indvars.iv.i, 1
  br label %for.body3.i
}

attributes #0 = { noreturn uwtable "frame-pointer"="all" }
