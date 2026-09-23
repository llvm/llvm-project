; RUN: opt %loadNPMPolly -polly-stmt-granularity=store '-passes=polly-custom<simplify-0;optree;delicm;simplify-1;opt-isl;ast>' -polly-print-ast -disable-output < %s | FileCheck %s
;
; Do not distribute the innermost point loop if the statements in it depend on
; each other in both directions.
;
;    void cyc(int n, double A[restrict n][n], double B[restrict n][n]) {
;      for (int i = 1; i < n; i++)
;        for (int j = 1; j < n; j++) {
;          A[i][j] = B[i][j - 1] + A[i - 1][j];
;          B[i][j] = 2 * A[i][j];
;        }
;    }
;
; The second statement depends on the first within an iteration of j, and the
; first on the second from the previous iteration of j.

; CHECK:      // 1st level tiling - Points
; CHECK-NEXT: for (int c2 = {{.*}})
; CHECK-NEXT:   for (int c3 = {{.*}}) {
; CHECK-NEXT:     Stmt_for_body4(
; CHECK-NEXT:     Stmt_for_body4_b(
; CHECK-NEXT:   }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @cyc(i32 %n, ptr noalias %A, ptr noalias %B) {
entry:
  %0 = zext i32 %n to i64
  %cmp49 = icmp sgt i32 %n, 1
  br i1 %cmp49, label %for.cond1.preheader.preheader, label %for.cond.cleanup

for.cond1.preheader.preheader:
  %wide.trip.count56 = zext nneg i32 %n to i64
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv52 = phi i64 [ 1, %for.cond1.preheader.preheader ], [ %indvars.iv.next53, %for.cond1.for.cond.cleanup3_crit_edge ]
  %1 = mul nuw nsw i64 %indvars.iv52, %0
  %arrayidx = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %1
  %2 = add nsw i64 %indvars.iv52, -1
  %3 = mul nuw nsw i64 %2, %0
  %arrayidx9 = getelementptr inbounds [8 x i8], ptr %A, i64 %3
  %arrayidx13 = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %1
  %load_initial = load double, ptr %arrayidx, align 8
  br label %for.body4

for.cond.cleanup:
  ret void

for.cond1.for.cond.cleanup3_crit_edge:
  %indvars.iv.next53 = add nuw nsw i64 %indvars.iv52, 1
  %exitcond57.not = icmp eq i64 %indvars.iv.next53, %wide.trip.count56
  br i1 %exitcond57.not, label %for.cond.cleanup, label %for.cond1.preheader

for.body4:
  %store_forwarded = phi double [ %load_initial, %for.cond1.preheader ], [ %mul, %for.body4 ]
  %indvars.iv = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %4 = getelementptr [8 x i8], ptr %arrayidx, i64 %indvars.iv
  %arrayidx11 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx9, i64 %indvars.iv
  %5 = load double, ptr %arrayidx11, align 8
  %add = fadd double %store_forwarded, %5
  %arrayidx15 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx13, i64 %indvars.iv
  store double %add, ptr %arrayidx15, align 8
  %mul = fmul double %add, 2.000000e+00
  store double %mul, ptr %4, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond1.for.cond.cleanup3_crit_edge, label %for.body4
}
