; RUN: opt %loadNPMPolly '-passes=polly-custom<optree;opt-isl>' -plugin-arg=Polly,-polly-print-opt-isl -disable-output < %s | FileCheck %s
;
; Compute the schedule from the exact dependences if the scheduler finds no
; schedule for the simplified ones.
;
;    void update_solve(int n, double A[restrict n][n], double x[restrict n],
;                      double y[restrict n]) {
;      for (int i = 0; i < n; i++)
;        for (int j = 0; j < n; j++)
;          A[i][j] += x[i] * y[j];
;      for (int i = n - 1; i >= 0; i--) {
;        double w = y[i];
;        for (int j = i + 1; j < n; j++)
;          w -= A[i][j] * x[j];
;        x[i] = w / A[i][i];
;      }
;    }
;
; The scheduler finds no schedule for the simplified dependences of this
; function.

; CHECK:      Calculated schedule:
; CHECK-NEXT: domain:
; CHECK:      mark: "1st level tiling - Tiles"
; CHECK-NEXT: child:
; CHECK-NEXT:   schedule: "[n] -> [{ Stmt_for_body4[i0, i1] -> [(floor((i0)/32))] }, { Stmt_for_body4[i0, i1] -> [(floor((i1)/32))] }]"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @update_solve(i32 %n, ptr noalias %A, ptr noalias %x, ptr noalias %y) {
entry:
  %0 = zext i32 %n to i64
  %cmp74 = icmp sgt i32 %n, 0
  br i1 %cmp74, label %for.cond1.preheader, label %for.cond.cleanup17

for.cond1.preheader:
  %indvars.iv84 = phi i64 [ %indvars.iv.next85, %for.cond1.for.cond.cleanup3_crit_edge ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw [8 x i8], ptr %x, i64 %indvars.iv84
  %1 = load double, ptr %arrayidx, align 8
  %2 = mul nuw nsw i64 %indvars.iv84, %0
  %arrayidx8 = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %2
  br label %for.body4

for.cond15.preheader:
  br label %for.body18.preheader

for.body18.preheader:
  %3 = zext nneg i32 %n to i64
  br label %for.body18

for.cond1.for.cond.cleanup3_crit_edge:
  %indvars.iv.next85 = add nuw nsw i64 %indvars.iv84, 1
  %exitcond88.not = icmp eq i64 %indvars.iv.next85, %0
  br i1 %exitcond88.not, label %for.cond15.preheader, label %for.cond1.preheader

for.body4:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %arrayidx6 = getelementptr inbounds nuw [8 x i8], ptr %y, i64 %indvars.iv
  %4 = load double, ptr %arrayidx6, align 8
  %arrayidx10 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx8, i64 %indvars.iv
  %5 = load double, ptr %arrayidx10, align 8
  %6 = tail call double @llvm.fmuladd.f64(double %1, double %4, double %5)
  store double %6, ptr %arrayidx10, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %0
  br i1 %exitcond.not, label %for.cond1.for.cond.cleanup3_crit_edge, label %for.body4

for.cond.cleanup17:
  ret void

for.body18:
  %indvars.iv89 = phi i64 [ %0, %for.body18.preheader ], [ %indvars.iv.next90, %for.cond.cleanup24 ]
  %indvars.iv.next90 = add nsw i64 %indvars.iv89, -1
  %arrayidx20 = getelementptr inbounds nuw [8 x i8], ptr %y, i64 %indvars.iv.next90
  %7 = load double, ptr %arrayidx20, align 8
  %cmp2376 = icmp slt i64 %indvars.iv89, %3
  %8 = mul nuw nsw i64 %indvars.iv.next90, %0
  br i1 %cmp2376, label %for.body25.lr.ph, label %for.cond.cleanup24

for.body25.lr.ph:
  %arrayidx27 = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %8
  br label %for.body25

for.cond.cleanup24:
  %w.0.lcssa = phi double [ %7, %for.body18 ], [ %12, %for.body25 ]
  %arrayidx36 = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %8
  %arrayidx38 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx36, i64 %indvars.iv.next90
  %9 = load double, ptr %arrayidx38, align 8
  %div = fdiv double %w.0.lcssa, %9
  %arrayidx40 = getelementptr inbounds nuw [8 x i8], ptr %x, i64 %indvars.iv.next90
  store double %div, ptr %arrayidx40, align 8
  %cmp16 = icmp samesign ugt i64 %indvars.iv89, 1
  br i1 %cmp16, label %for.body18, label %for.cond.cleanup17

for.body25:
  %indvars.iv91 = phi i64 [ %indvars.iv89, %for.body25.lr.ph ], [ %indvars.iv.next92, %for.body25 ]
  %w.077 = phi double [ %7, %for.body25.lr.ph ], [ %12, %for.body25 ]
  %arrayidx29 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx27, i64 %indvars.iv91
  %10 = load double, ptr %arrayidx29, align 8
  %arrayidx31 = getelementptr inbounds nuw [8 x i8], ptr %x, i64 %indvars.iv91
  %11 = load double, ptr %arrayidx31, align 8
  %neg = fneg double %10
  %12 = tail call double @llvm.fmuladd.f64(double %neg, double %11, double %w.077)
  %indvars.iv.next92 = add nuw nsw i64 %indvars.iv91, 1
  %13 = trunc nuw i64 %indvars.iv.next92 to i32
  %cmp23 = icmp sgt i32 %n, %13
  br i1 %cmp23, label %for.body25, label %for.cond.cleanup24
}

declare double @llvm.fmuladd.f64(double, double, double)
