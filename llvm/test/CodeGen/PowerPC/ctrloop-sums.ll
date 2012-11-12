; ModuleID = 'SingleSource/Regression/C/sumarray2d.c'
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc < %s -march=ppc64 | FileCheck %s

@.str = private unnamed_addr constant [23 x i8] c"Sum(Array[%d,%d] = %d\0A\00", align 1

define i32 @SumArray([100 x i32]* nocapture %Array, i32 %NumI, i32 %NumJ) nounwind readonly {
entry:
  %cmp12 = icmp eq i32 %NumI, 0
  br i1 %cmp12, label %for.end8, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp29 = icmp eq i32 %NumJ, 0
  br i1 %cmp29, label %for.inc6, label %for.body3.lr.ph.us

for.inc6.us:                                      ; preds = %for.body3.us
  %indvars.iv.next17 = add i64 %indvars.iv16, 1
  %lftr.wideiv18 = trunc i64 %indvars.iv.next17 to i32
  %exitcond19 = icmp eq i32 %lftr.wideiv18, %NumI
  br i1 %exitcond19, label %for.end8, label %for.body3.lr.ph.us

for.body3.us:                                     ; preds = %for.body3.us, %for.body3.lr.ph.us
  %indvars.iv = phi i64 [ 0, %for.body3.lr.ph.us ], [ %indvars.iv.next, %for.body3.us ]
  %Result.111.us = phi i32 [ %Result.014.us, %for.body3.lr.ph.us ], [ %add.us, %for.body3.us ]
  %arrayidx5.us = getelementptr inbounds [100 x i32]* %Array, i64 %indvars.iv16, i64 %indvars.iv
  %0 = load i32* %arrayidx5.us, align 4, !tbaa !0
  %add.us = add nsw i32 %0, %Result.111.us
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %NumJ
  br i1 %exitcond, label %for.inc6.us, label %for.body3.us

for.body3.lr.ph.us:                               ; preds = %for.inc6.us, %for.cond1.preheader.lr.ph
  %indvars.iv16 = phi i64 [ %indvars.iv.next17, %for.inc6.us ], [ 0, %for.cond1.preheader.lr.ph ]
  %Result.014.us = phi i32 [ %add.us, %for.inc6.us ], [ 0, %for.cond1.preheader.lr.ph ]
  br label %for.body3.us

for.inc6:                                         ; preds = %for.inc6, %for.cond1.preheader.lr.ph
  %i.013 = phi i32 [ %inc7, %for.inc6 ], [ 0, %for.cond1.preheader.lr.ph ]
  %inc7 = add i32 %i.013, 1
  %exitcond20 = icmp eq i32 %inc7, %NumI
  br i1 %exitcond20, label %for.end8, label %for.inc6

for.end8:                                         ; preds = %for.inc6.us, %for.inc6, %entry
  %Result.0.lcssa = phi i32 [ 0, %entry ], [ %add.us, %for.inc6.us ], [ 0, %for.inc6 ]
  ret i32 %Result.0.lcssa
; CHECK: @SumArray
; CHECK: mtctr
; CHECK: bdnz
}

define i32 @main() nounwind {
entry:
  %Array = alloca [100 x [100 x i32]], align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv33 = phi i64 [ 0, %entry ], [ %indvars.iv.next34, %for.body ]
  %0 = trunc i64 %indvars.iv33 to i32
  %sub = sub i32 0, %0
  %arrayidx2 = getelementptr inbounds [100 x [100 x i32]]* %Array, i64 0, i64 %indvars.iv33, i64 %indvars.iv33
  store i32 %sub, i32* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next34 = add i64 %indvars.iv33, 1
  %lftr.wideiv35 = trunc i64 %indvars.iv.next34 to i32
  %exitcond36 = icmp eq i32 %lftr.wideiv35, 100
  br i1 %exitcond36, label %for.cond6.preheader, label %for.body

for.cond6.preheader:                              ; preds = %for.body, %for.inc17
  %indvars.iv29 = phi i64 [ %indvars.iv.next30, %for.inc17 ], [ 0, %for.body ]
  br label %for.body8

for.body8:                                        ; preds = %for.inc14, %for.cond6.preheader
  %indvars.iv = phi i64 [ 0, %for.cond6.preheader ], [ %indvars.iv.next, %for.inc14 ]
  %1 = trunc i64 %indvars.iv to i32
  %2 = trunc i64 %indvars.iv29 to i32
  %cmp9 = icmp eq i32 %1, %2
  br i1 %cmp9, label %for.inc14, label %if.then

if.then:                                          ; preds = %for.body8
  %3 = add i64 %indvars.iv, %indvars.iv29
  %arrayidx13 = getelementptr inbounds [100 x [100 x i32]]* %Array, i64 0, i64 %indvars.iv29, i64 %indvars.iv
  %4 = trunc i64 %3 to i32
  store i32 %4, i32* %arrayidx13, align 4, !tbaa !0
  br label %for.inc14

for.inc14:                                        ; preds = %for.body8, %if.then
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv27 = trunc i64 %indvars.iv.next to i32
  %exitcond28 = icmp eq i32 %lftr.wideiv27, 100
  br i1 %exitcond28, label %for.inc17, label %for.body8

for.inc17:                                        ; preds = %for.inc14
  %indvars.iv.next30 = add i64 %indvars.iv29, 1
  %lftr.wideiv31 = trunc i64 %indvars.iv.next30 to i32
  %exitcond32 = icmp eq i32 %lftr.wideiv31, 100
  br i1 %exitcond32, label %for.body3.lr.ph.us.i, label %for.cond6.preheader

for.inc6.us.i:                                    ; preds = %for.body3.us.i
  %indvars.iv.next17.i = add i64 %indvars.iv16.i, 1
  %lftr.wideiv24 = trunc i64 %indvars.iv.next17.i to i32
  %exitcond25 = icmp eq i32 %lftr.wideiv24, 100
  br i1 %exitcond25, label %SumArray.exit, label %for.body3.lr.ph.us.i

for.body3.us.i:                                   ; preds = %for.body3.lr.ph.us.i, %for.body3.us.i
  %indvars.iv.i = phi i64 [ 0, %for.body3.lr.ph.us.i ], [ %indvars.iv.next.i, %for.body3.us.i ]
  %Result.111.us.i = phi i32 [ %Result.014.us.i, %for.body3.lr.ph.us.i ], [ %add.us.i, %for.body3.us.i ]
  %arrayidx5.us.i = getelementptr inbounds [100 x [100 x i32]]* %Array, i64 0, i64 %indvars.iv16.i, i64 %indvars.iv.i
  %5 = load i32* %arrayidx5.us.i, align 4, !tbaa !0
  %add.us.i = add nsw i32 %5, %Result.111.us.i
  %indvars.iv.next.i = add i64 %indvars.iv.i, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next.i to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 100
  br i1 %exitcond, label %for.inc6.us.i, label %for.body3.us.i

for.body3.lr.ph.us.i:                             ; preds = %for.inc17, %for.inc6.us.i
  %indvars.iv16.i = phi i64 [ %indvars.iv.next17.i, %for.inc6.us.i ], [ 0, %for.inc17 ]
  %Result.014.us.i = phi i32 [ %add.us.i, %for.inc6.us.i ], [ 0, %for.inc17 ]
  br label %for.body3.us.i

SumArray.exit:                                    ; preds = %for.inc6.us.i
  %call20 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([23 x i8]* @.str, i64 0, i64 0), i32 100, i32 100, i32 %add.us.i) nounwind
  ret i32 0

; CHECK: @main
; CHECK: mtctr
; CHECK: bdnz
}

declare i32 @printf(i8* nocapture, ...) nounwind

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
