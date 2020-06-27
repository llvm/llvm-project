; RUN: opt < %s -basic-aa -loop-interchange -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa -stats 2>&1 | FileCheck %s
; RUN: FileCheck --input-file=%t --check-prefix=REMARKS %s


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; REMARKS: --- !Passed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            Interchanged
; REMARKS-NEXT: Function:        test1

define i64 @test1([100 x [100 x i64]]* %Arr) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[FOR2_PREHEADER:%.*]]
; CHECK:       for1.header.preheader:
; CHECK-NEXT:    br label [[FOR1_HEADER:%.*]]
; CHECK:       for1.header:
; CHECK-NEXT:    [[INDVARS_IV23:%.*]] = phi i64 [ [[INDVARS_IV_NEXT24:%.*]], [[FOR1_INC:%.*]] ], [ 0, [[FOR1_HEADER_PREHEADER:%.*]] ]
; CHECK-NEXT:    [[SUM_INNER:%.*]] = phi i64 [ [[SUM_INC:%.*]], [[FOR1_INC]] ], [ [[SUM_OUTER:%.*]], [[FOR1_HEADER_PREHEADER]] ]
; CHECK-NEXT:    br label [[FOR2_SPLIT1:%.*]]
; CHECK:       for2.preheader:
; CHECK-NEXT:    br label [[FOR2:%.*]]
; CHECK:       for2:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_NEXT_3:%.*]], [[FOR2_SPLIT:%.*]] ], [ 0, [[FOR2_PREHEADER]] ]
; CHECK-NEXT:    [[SUM_OUTER]] = phi i64 [ [[SUM_INC_LCSSA:%.*]], [[FOR2_SPLIT]] ], [ 0, [[FOR2_PREHEADER]] ]
; CHECK-NEXT:    br label [[FOR1_HEADER_PREHEADER]]
; CHECK:       for2.split1:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* [[ARR:%.*]], i64 0, i64 [[INDVARS_IV]], i64 [[INDVARS_IV23]]
; CHECK-NEXT:    [[LV:%.*]] = load i64, i64* [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[SUM_INC]] = add i64 [[SUM_INNER]], [[LV]]
; CHECK-NEXT:    [[IV_ORIGINAL:%.*]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXIT1_ORIGINAL:%.*]] = icmp eq i64 [[IV_ORIGINAL]], 100
; CHECK-NEXT:    br label [[FOR1_INC]]
; CHECK:       for2.split:
; CHECK-NEXT:    [[SUM_INC_LCSSA]] = phi i64 [ [[SUM_INC]], %for1.inc ]
; CHECK-NEXT:    [[INDVARS_IV_NEXT_3]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXIT1:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT_3]], 100
; CHECK-NEXT:    br i1 [[EXIT1]], label [[FOR1_LOOPEXIT:%.*]], label [[FOR2]]
; CHECK:       for1.inc:
; CHECK-NEXT:    [[INDVARS_IV_NEXT24]] = add nuw nsw i64 [[INDVARS_IV23]], 1
; CHECK-NEXT:    [[EXIT2:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT24]], 100
; CHECK-NEXT:    br i1 [[EXIT2]], label [[FOR2_SPLIT]], label [[FOR1_HEADER]]
; CHECK:       for1.loopexit:
; CHECK-NEXT:    [[SUM_INC_LCSSA2:%.*]] = phi i64 [ [[SUM_INC_LCSSA]], [[FOR2_SPLIT]] ]
; CHECK-NEXT:    ret i64 [[SUM_INC_LCSSA2]]
;
entry:
  br label %for1.header

for1.header:                                         ; preds = %for1.inc, %entry
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for1.inc ]
  %sum.outer = phi i64 [ 0, %entry ], [ %sum.inc.lcssa, %for1.inc ]
  br label %for2

for2:                                        ; preds = %for2, %for1.header
  %indvars.iv = phi i64 [ 0, %for1.header ], [ %indvars.iv.next.3, %for2 ]
  %sum.inner = phi i64 [ %sum.outer, %for1.header ], [ %sum.inc, %for2 ]
  %arrayidx = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* %Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %lv = load i64, i64* %arrayidx, align 4
  %sum.inc = add i64 %sum.inner, %lv
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 1
  %exit1 = icmp eq i64 %indvars.iv.next.3, 100
  br i1 %exit1, label %for1.inc, label %for2

for1.inc:                                ; preds = %for2
  %sum.inc.lcssa = phi i64 [ %sum.inc, %for2 ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exit2 = icmp eq i64 %indvars.iv.next24, 100
  br i1 %exit2, label %for1.loopexit, label %for1.header

for1.loopexit:                                 ; preds = %for1.inc
  %sum.inc.lcssa2 = phi i64 [ %sum.inc.lcssa, %for1.inc ]
  ret i64 %sum.inc.lcssa2
}

; In this test case, the inner reduction PHI %inner does not involve the outer
; reduction PHI %sum.outer, do not interchange.
; REMARKS: --- !Missed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            UnsupportedPHIOuter
; REMARKS-NEXT: Function:        test2

define i64 @test2([100 x [100 x i64]]* %Arr) {
entry:
  br label %for1.header

for1.header:                                         ; preds = %for1.inc, %entry
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for1.inc ]
  %sum.outer = phi i64 [ 0, %entry ], [ %sum.inc.lcssa, %for1.inc ]
  br label %for2

for2:                                        ; preds = %for2, %for1.header
  %indvars.iv = phi i64 [ 0, %for1.header ], [ %indvars.iv.next.3, %for2 ]
  %inner = phi i64 [ %indvars.iv23, %for1.header ], [ %sum.inc, %for2 ]
  %arrayidx = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* %Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %lv = load i64, i64* %arrayidx, align 4
  %sum.inc = add i64 %inner, %lv
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 1
  %exit1 = icmp eq i64 %indvars.iv.next.3, 100
  br i1 %exit1, label %for1.inc, label %for2

for1.inc:                                ; preds = %for2
  %sum.inc.lcssa = phi i64 [ %sum.inc, %for2 ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exit2 = icmp eq i64 %indvars.iv.next24, 100
  br i1 %exit2, label %for1.loopexit, label %for1.header

for1.loopexit:                                 ; preds = %for1.inc
  %sum.inc.lcssa2 = phi i64 [ %sum.inc.lcssa, %for1.inc ]
  ret i64 %sum.inc.lcssa2
}

; Check that we do not interchange if there is an additional instruction
; between the outer and inner reduction PHIs.
; REMARKS: --- !Missed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            UnsupportedPHIOuter
; REMARKS-NEXT: Function:        test3

define i64 @test3([100 x [100 x i64]]* %Arr) {
entry:
  br label %for1.header

for1.header:                                         ; preds = %for1.inc, %entry
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for1.inc ]
  %sum.outer = phi i64 [ 0, %entry ], [ %sum.inc.lcssa, %for1.inc ]
  %so = add i64 %sum.outer, 10
  br label %for2

for2:                                        ; preds = %for2, %for1.header
  %indvars.iv = phi i64 [ 0, %for1.header ], [ %indvars.iv.next.3, %for2 ]
  %sum.inner = phi i64 [ %so, %for1.header ], [ %sum.inc, %for2 ]
  %arrayidx = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* %Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %lv = load i64, i64* %arrayidx, align 4
  %sum.inc = add i64 %sum.inner, %lv
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 1
  %exit1 = icmp eq i64 %indvars.iv.next.3, 100
  br i1 %exit1, label %for1.inc, label %for2

for1.inc:                                ; preds = %for2
  %sum.inc.lcssa = phi i64 [ %sum.inc, %for2 ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exit2 = icmp eq i64 %indvars.iv.next24, 100
  br i1 %exit2, label %for1.loopexit, label %for1.header

for1.loopexit:                                 ; preds = %for1.inc
  %sum.inc.lcssa2 = phi i64 [ %sum.inc.lcssa, %for1.inc ]
  ret i64 %sum.inc.lcssa2
}
