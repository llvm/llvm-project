; RUN: opt -consthoist -mtriple=x86_64-unknown-linux-gnu -consthoist-with-block-frequency=true -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Check when BFI is enabled for constant hoisting, constant 214748364701
; will not be hoisted to the func entry.
; CHECK-LABEL: @foo(
; CHECK: entry:
; CHECK-NOT: bitcast i64 214748364701 to i64
; CHECK: if.then:

; Function Attrs: norecurse nounwind uwtable
define i64 @foo(i64* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 9
  %t0 = load i64, i64* %arrayidx, align 8
  %cmp = icmp slt i64 %t0, 564
  br i1 %cmp, label %if.then, label %if.else5

if.then:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds i64, i64* %a, i64 5
  %t1 = load i64, i64* %arrayidx1, align 8
  %cmp2 = icmp slt i64 %t1, 1009
  br i1 %cmp2, label %if.then3, label %return

if.then3:                                         ; preds = %if.then
  %arrayidx4 = getelementptr inbounds i64, i64* %a, i64 6
  %t2 = load i64, i64* %arrayidx4, align 8
  %inc = add nsw i64 %t2, 1
  store i64 %inc, i64* %arrayidx4, align 8
  br label %return

if.else5:                                         ; preds = %entry
  %arrayidx6 = getelementptr inbounds i64, i64* %a, i64 6
  %t3 = load i64, i64* %arrayidx6, align 8
  %cmp7 = icmp slt i64 %t3, 3512
  br i1 %cmp7, label %if.then8, label %return

if.then8:                                         ; preds = %if.else5
  %arrayidx9 = getelementptr inbounds i64, i64* %a, i64 7
  %t4 = load i64, i64* %arrayidx9, align 8
  %inc10 = add nsw i64 %t4, 1
  store i64 %inc10, i64* %arrayidx9, align 8
  br label %return

return:                                           ; preds = %if.else5, %if.then, %if.then8, %if.then3
  %retval.0 = phi i64 [ 214748364701, %if.then3 ], [ 214748364701, %if.then8 ], [ 250148364702, %if.then ], [ 256148364704, %if.else5 ]
  ret i64 %retval.0
}

; Check when BFI is enabled for constant hoisting, constant 214748364701
; in while.body will be hoisted to while.body.preheader. 214748364701 in
; if.then16 and if.else10 will be merged and hoisted to the beginning of
; if.else10 because if.else10 dominates if.then16.
; CHECK-LABEL: @goo(
; CHECK: entry:
; CHECK-NOT: bitcast i64 214748364701 to i64
; CHECK: while.body.preheader:
; CHECK-NEXT: bitcast i64 214748364701 to i64
; CHECK-NOT: bitcast i64 214748364701 to i64
; CHECK: if.else10:
; CHECK-NEXT: bitcast i64 214748364701 to i64
; CHECK-NOT: bitcast i64 214748364701 to i64
define i64 @goo(i64* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 9
  %t0 = load i64, i64* %arrayidx, align 8
  %cmp = icmp ult i64 %t0, 56
  br i1 %cmp, label %if.then, label %if.else10, !prof !0

if.then:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds i64, i64* %a, i64 5
  %t1 = load i64, i64* %arrayidx1, align 8
  %cmp2 = icmp ult i64 %t1, 10
  br i1 %cmp2, label %while.cond.preheader, label %return, !prof !0

while.cond.preheader:                             ; preds = %if.then
  %arrayidx7 = getelementptr inbounds i64, i64* %a, i64 6
  %t2 = load i64, i64* %arrayidx7, align 8
  %cmp823 = icmp ugt i64 %t2, 10000
  br i1 %cmp823, label %while.body.preheader, label %return

while.body.preheader:                             ; preds = %while.cond.preheader
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %t3 = phi i64 [ %add, %while.body ], [ %t2, %while.body.preheader ]
  %add = add i64 %t3, 214748364701
  %cmp8 = icmp ugt i64 %add, 10000
  br i1 %cmp8, label %while.body, label %while.cond.return.loopexit_crit_edge

if.else10:                                        ; preds = %entry
  %arrayidx11 = getelementptr inbounds i64, i64* %a, i64 6
  %t4 = load i64, i64* %arrayidx11, align 8
  %add2 = add i64 %t4, 214748364701
  %cmp12 = icmp ult i64 %add2, 35
  br i1 %cmp12, label %if.then16, label %return, !prof !0

if.then16:                                        ; preds = %if.else10
  %arrayidx17 = getelementptr inbounds i64, i64* %a, i64 7
  %t5 = load i64, i64* %arrayidx17, align 8
  %inc = add i64 %t5, 1
  store i64 %inc, i64* %arrayidx17, align 8
  br label %return

while.cond.return.loopexit_crit_edge:             ; preds = %while.body
  store i64 %add, i64* %arrayidx7, align 8
  br label %return

return:                                           ; preds = %while.cond.preheader, %while.cond.return.loopexit_crit_edge, %if.else10, %if.then, %if.then16
  %retval.0 = phi i64 [ 214748364701, %if.then16 ], [ 0, %if.then ], [ 0, %if.else10 ], [ 0, %while.cond.return.loopexit_crit_edge ], [ 0, %while.cond.preheader ]
  ret i64 %retval.0
}

!0 = !{!"branch_weights", i32 1, i32 2000}
