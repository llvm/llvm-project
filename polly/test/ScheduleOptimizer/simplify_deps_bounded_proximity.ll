; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl>' -polly-print-opt-isl -disable-output < %s | FileCheck %s
;
; Keep the exact proximity dependence of a statement on itself if simplifying
; it makes its distances unbounded.
;
;    void floyd(int n, int path[restrict n][n]) {
;      for (int k = 0; k < n; k++)
;        for (int i = 0; i < n; i++)
;          for (int j = 0; j < n; j++)
;            path[i][j] = path[i][j] < path[i][k] + path[k][j]
;                             ? path[i][j]
;                             : path[i][k] + path[k][j];
;    }
;
; Within an iteration of k, path[i][k] is read by all later iterations of j
; and path[k][j] by all later iterations of i. The simplified dependences lose
; the upper bound of j and i, so no schedule row along i or j bounds their
; distance. The scheduler then carried the dependences with a wavefront over
; i + j instead of forming a permutable band of i and j.

; CHECK:      schedule: "[n] -> [{ Stmt_for_body8[i0, i1, i2] -> [(i0)] }]"
; CHECK-NEXT: permutable: 1
; CHECK-NEXT: child:
; CHECK-NEXT:   mark: "1st level tiling - Tiles"
; CHECK-NEXT:   child:
; CHECK-NEXT:     schedule: "[n] -> [{ Stmt_for_body8[i0, i1, i2] -> [(floor((i1)/32))] }, { Stmt_for_body8[i0, i1, i2] -> [(floor((i2)/32))] }]"
; CHECK-NEXT:     permutable: 1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @floyd(i32 %n, ptr noalias %path) {
entry:
  %0 = zext i32 %n to i64
  %cmp74 = icmp sgt i32 %n, 0
  br i1 %cmp74, label %for.cond1.preheader.preheader, label %for.cond.cleanup

for.cond1.preheader.preheader:
  %wide.trip.count85 = zext nneg i32 %n to i64
  %wide.trip.count80 = zext nneg i32 %n to i64
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv82 = phi i64 [ 0, %for.cond1.preheader.preheader ], [ %indvars.iv.next83, %for.cond1.for.cond.cleanup3_crit_edge ]
  %1 = mul nuw nsw i64 %indvars.iv82, %0
  %arrayidx16 = getelementptr inbounds nuw [4 x i8], ptr %path, i64 %1
  br label %for.cond5.preheader

for.cond.cleanup:
  ret void

for.cond5.preheader:
  %indvars.iv77 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next78, %for.cond5.for.cond.cleanup7_crit_edge ]
  %2 = mul nuw nsw i64 %indvars.iv77, %0
  %arrayidx = getelementptr inbounds nuw [4 x i8], ptr %path, i64 %2
  %arrayidx14 = getelementptr inbounds nuw [4 x i8], ptr %arrayidx, i64 %indvars.iv82
  br label %for.body8

for.cond1.for.cond.cleanup3_crit_edge:
  %indvars.iv.next83 = add nuw nsw i64 %indvars.iv82, 1
  %exitcond86.not = icmp eq i64 %indvars.iv.next83, %wide.trip.count85
  br i1 %exitcond86.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond5.for.cond.cleanup7_crit_edge:
  %indvars.iv.next78 = add nuw nsw i64 %indvars.iv77, 1
  %exitcond81.not = icmp eq i64 %indvars.iv.next78, %wide.trip.count80
  br i1 %exitcond81.not, label %for.cond1.for.cond.cleanup3_crit_edge, label %for.cond5.preheader

for.body8:
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx10 = getelementptr inbounds nuw [4 x i8], ptr %arrayidx, i64 %indvars.iv
  %3 = load i32, ptr %arrayidx10, align 4
  %4 = load i32, ptr %arrayidx14, align 4
  %arrayidx18 = getelementptr inbounds nuw [4 x i8], ptr %arrayidx16, i64 %indvars.iv
  %5 = load i32, ptr %arrayidx18, align 4
  %add = add nsw i32 %5, %4
  %.add = tail call i32 @llvm.smin.i32(i32 %3, i32 %add)
  store i32 %.add, ptr %arrayidx10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond5.for.cond.cleanup7_crit_edge, label %for.body8
}

declare i32 @llvm.smin.i32(i32, i32)
