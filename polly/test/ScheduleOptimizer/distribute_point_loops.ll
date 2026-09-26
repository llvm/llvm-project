; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -polly-print-ast -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -polly-print-ast -polly-distribute-point-loops=false -disable-output < %s | FileCheck %s --check-prefix=FUSED
;
; Distribute the innermost point loop of a time tiled stencil over its
; statements.
;
;    void jacobi(int tsteps, int n, double A[restrict n][n],
;                double B[restrict n][n]) {
;      for (int t = 0; t < tsteps; t++) {
;        for (int i = 1; i < n - 1; i++)
;          for (int j = 1; j < n - 1; j++)
;            B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][j + 1] +
;                             A[i + 1][j] + A[i - 1][j]);
;        for (int i = 1; i < n - 1; i++)
;          for (int j = 1; j < n - 1; j++)
;            A[i][j] = 0.2 * (B[i][j] + B[i][j - 1] + B[i][j + 1] +
;                             B[i + 1][j] + B[i - 1][j]);
;      }
;    }
;
; The scheduler skews the loops over i and j by 2t and shifts the second
; statement by one iteration against the first, so that both form a single
; permutable band. The innermost point loop then runs both statements under
; conditions and carries the dependence of the second statement on the first.
; Within an iteration of the loops around it, however, only the second
; statement depends on the first, and neither loop carries a dependence on
; its own, so the loop is run once for each statement.

; CHECK:      // 1st level tiling - Points
; CHECK-NEXT: for (int c3 = {{.*}})
; CHECK-NEXT:   for (int c4 = {{.*}}) {
; CHECK-NEXT:     if ({{.*}})
; CHECK-NEXT:       for (int c5 = {{.*}})
; CHECK-NEXT:         Stmt_for_body9(
; CHECK-NEXT:     if ({{.*}})
; CHECK-NEXT:       for (int c5 = {{.*}})
; CHECK-NEXT:         Stmt_for_body53(
; CHECK-NEXT:   }

; FUSED:      // 1st level tiling - Points
; FUSED-NEXT: for (int c3 = {{.*}})
; FUSED-NEXT:   for (int c4 = {{.*}})
; FUSED-NEXT:     for (int c5 = {{.*}}) {
; FUSED-NEXT:       if ({{.*}})
; FUSED-NEXT:         Stmt_for_body9(
; FUSED-NEXT:       if ({{.*}})
; FUSED-NEXT:         Stmt_for_body53(
; FUSED-NEXT:     }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @jacobi(i32 %tsteps, i32 %n, ptr noalias %A, ptr noalias %B) {
entry:
  %0 = zext i32 %n to i64
  %cmp150 = icmp sgt i32 %tsteps, 0
  br i1 %cmp150, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:
  %sub = add i32 %n, -1
  %cmp2144 = icmp sgt i32 %n, 2
  %cmp45148 = icmp sgt i32 %n, 2
  %wide.trip.count157 = zext i32 %sub to i64
  %wide.trip.count = zext i32 %sub to i64
  %wide.trip.count168 = zext i32 %sub to i64
  %wide.trip.count162 = zext i32 %sub to i64
  br label %for.cond1.preheader

for.cond1.preheader:
  %t.0151 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc94, %for.cond.cleanup46 ]
  br i1 %cmp2144, label %for.cond5.preheader, label %for.cond43.preheader

for.cond.cleanup:
  ret void

for.cond43.preheader:
  br i1 %cmp45148, label %for.cond49.preheader, label %for.cond.cleanup46

for.cond5.preheader:
  %indvars.iv153 = phi i64 [ %indvars.iv.next154, %for.cond5.for.cond.cleanup8_crit_edge ], [ 1, %for.cond1.preheader ]
  %1 = mul nuw nsw i64 %indvars.iv153, %0
  %arrayidx = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %1
  %indvars.iv.next154 = add nuw nsw i64 %indvars.iv153, 1
  %2 = mul nuw nsw i64 %indvars.iv.next154, %0
  %arrayidx25 = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %2
  %3 = add nsw i64 %indvars.iv153, -1
  %4 = mul nuw nsw i64 %3, %0
  %arrayidx31 = getelementptr inbounds [8 x i8], ptr %A, i64 %4
  %arrayidx36 = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %1
  br label %for.body9

for.cond5.for.cond.cleanup8_crit_edge:
  %exitcond158.not = icmp eq i64 %indvars.iv.next154, %wide.trip.count157
  br i1 %exitcond158.not, label %for.cond43.preheader, label %for.cond5.preheader

for.body9:
  %indvars.iv = phi i64 [ 1, %for.cond5.preheader ], [ %indvars.iv.next, %for.body9 ]
  %arrayidx11 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx, i64 %indvars.iv
  %5 = load double, ptr %arrayidx11, align 8
  %arrayidx16 = getelementptr i8, ptr %arrayidx11, i64 -8
  %6 = load double, ptr %arrayidx16, align 8
  %add = fadd double %5, %6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx21 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx, i64 %indvars.iv.next
  %7 = load double, ptr %arrayidx21, align 8
  %add22 = fadd double %add, %7
  %arrayidx27 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx25, i64 %indvars.iv
  %8 = load double, ptr %arrayidx27, align 8
  %add28 = fadd double %add22, %8
  %arrayidx33 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx31, i64 %indvars.iv
  %9 = load double, ptr %arrayidx33, align 8
  %add34 = fadd double %add28, %9
  %mul = fmul double %add34, 2.000000e-01
  %arrayidx38 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx36, i64 %indvars.iv
  store double %mul, ptr %arrayidx38, align 8
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond5.for.cond.cleanup8_crit_edge, label %for.body9

for.cond49.preheader:
  %indvars.iv164 = phi i64 [ %indvars.iv.next165, %for.cond49.for.cond.cleanup52_crit_edge ], [ 1, %for.cond43.preheader ]
  %10 = mul nuw nsw i64 %indvars.iv164, %0
  %arrayidx55 = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %10
  %indvars.iv.next165 = add nuw nsw i64 %indvars.iv164, 1
  %11 = mul nuw nsw i64 %indvars.iv.next165, %0
  %arrayidx72 = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %11
  %12 = add nsw i64 %indvars.iv164, -1
  %13 = mul nuw nsw i64 %12, %0
  %arrayidx78 = getelementptr inbounds [8 x i8], ptr %B, i64 %13
  %arrayidx84 = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %10
  br label %for.body53

for.cond.cleanup46:
  %inc94 = add nuw nsw i32 %t.0151, 1
  %exitcond170.not = icmp eq i32 %inc94, %tsteps
  br i1 %exitcond170.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond49.for.cond.cleanup52_crit_edge:
  %exitcond169.not = icmp eq i64 %indvars.iv.next165, %wide.trip.count168
  br i1 %exitcond169.not, label %for.cond.cleanup46, label %for.cond49.preheader

for.body53:
  %indvars.iv159 = phi i64 [ 1, %for.cond49.preheader ], [ %indvars.iv.next160, %for.body53 ]
  %arrayidx57 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx55, i64 %indvars.iv159
  %14 = load double, ptr %arrayidx57, align 8
  %arrayidx62 = getelementptr i8, ptr %arrayidx57, i64 -8
  %15 = load double, ptr %arrayidx62, align 8
  %add63 = fadd double %14, %15
  %indvars.iv.next160 = add nuw nsw i64 %indvars.iv159, 1
  %arrayidx68 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx55, i64 %indvars.iv.next160
  %16 = load double, ptr %arrayidx68, align 8
  %add69 = fadd double %add63, %16
  %arrayidx74 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx72, i64 %indvars.iv159
  %17 = load double, ptr %arrayidx74, align 8
  %add75 = fadd double %add69, %17
  %arrayidx80 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx78, i64 %indvars.iv159
  %18 = load double, ptr %arrayidx80, align 8
  %add81 = fadd double %add75, %18
  %mul82 = fmul double %add81, 2.000000e-01
  %arrayidx86 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx84, i64 %indvars.iv159
  store double %mul82, ptr %arrayidx86, align 8
  %exitcond163.not = icmp eq i64 %indvars.iv.next160, %wide.trip.count162
  br i1 %exitcond163.not, label %for.cond49.for.cond.cleanup52_crit_edge, label %for.body53
}
