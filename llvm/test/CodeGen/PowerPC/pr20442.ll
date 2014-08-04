; RUN: llc -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-linux-gnu"

; This code would cause code generation like this after PPCCTRLoops ran:
;  %indvar = phi i32 [ 0, %for.body ], [ %indvar.next, %if.then6 ]
;  %j.1.ph13 = phi i32 [ %j.110, %if.then6 ], [ 0, %for.body ], [ 0, %for.body ]
;  %c.0.ph12 = phi i32 [ %dec, %if.then6 ], [ %2, %for.body ], [ %2, %for.body ]
; which would fail verification because the created induction variable does not
; have as many predecessor entries as the other PHIs.
; CHECK-LABEL: @fn1
; CHECK: mtctr

%struct.anon = type { i32 }
%struct.anon.0 = type { i32 }

@b = common global %struct.anon* null, align 4
@a = common global %struct.anon.0* null, align 4

; Function Attrs: nounwind readonly uwtable
define i32 @fn1() #0 {
entry:
  %0 = load %struct.anon** @b, align 4
  %1 = ptrtoint %struct.anon* %0 to i32
  %cmp = icmp sgt %struct.anon* %0, null
  %2 = load %struct.anon.0** @a, align 4
  br i1 %cmp, label %for.bodythread-pre-split, label %if.end8

for.bodythread-pre-split:                         ; preds = %entry
  %aclass = getelementptr inbounds %struct.anon.0* %2, i32 0, i32 0
  %.pr = load i32* %aclass, align 4
  br label %for.body

for.body:                                         ; preds = %for.bodythread-pre-split, %for.body
  switch i32 %.pr, label %for.body [
    i32 0, label %while.body.lr.ph.preheader
    i32 2, label %while.body.lr.ph.preheader
  ]

while.body.lr.ph.preheader:                       ; preds = %for.body, %for.body
  br label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %while.body.lr.ph.preheader, %if.then6
  %j.1.ph13 = phi i32 [ %j.110.lcssa, %if.then6 ], [ 0, %while.body.lr.ph.preheader ]
  %c.0.ph12 = phi i32 [ %dec, %if.then6 ], [ %1, %while.body.lr.ph.preheader ]
  br label %while.body

while.cond:                                       ; preds = %while.body
  %cmp2 = icmp slt i32 %inc7, %c.0.ph12
  br i1 %cmp2, label %while.body, label %if.end8.loopexit

while.body:                                       ; preds = %while.body.lr.ph, %while.cond
  %j.110 = phi i32 [ %j.1.ph13, %while.body.lr.ph ], [ %inc7, %while.cond ]
  %aclass_index = getelementptr inbounds %struct.anon* %0, i32 %j.110, i32 0
  %3 = load i32* %aclass_index, align 4
  %aclass5 = getelementptr inbounds %struct.anon.0* %2, i32 %3, i32 0
  %4 = load i32* %aclass5, align 4
  %tobool = icmp eq i32 %4, 0
  %inc7 = add nsw i32 %j.110, 1
  br i1 %tobool, label %while.cond, label %if.then6

if.then6:                                         ; preds = %while.body
  %j.110.lcssa = phi i32 [ %j.110, %while.body ]
  %dec = add nsw i32 %c.0.ph12, -1
  %cmp29 = icmp slt i32 %j.110.lcssa, %dec
  br i1 %cmp29, label %while.body.lr.ph, label %if.end8.loopexit17

if.end8.loopexit:                                 ; preds = %while.cond
  br label %if.end8

if.end8.loopexit17:                               ; preds = %if.then6
  br label %if.end8

if.end8:                                          ; preds = %if.end8.loopexit17, %if.end8.loopexit, %entry
  ret i32 undef
}

attributes #0 = { nounwind readonly uwtable }

