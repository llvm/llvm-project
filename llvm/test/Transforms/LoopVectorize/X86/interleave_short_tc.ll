; Check that we won't interleave by more than half the "best known" estimated trip count.

;
; The loop is expected to be vectorized by 4 and computed interleaving factor is 1.
; Thus the resulting step is 4.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 -vectorizer-min-trip-count=4 -S < %s |  FileCheck %s

; The loop is expected to be vectorized by 2 and computed interleaving factor is 2.
; Thus the resulting step is 4.
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -vectorizer-min-trip-count=4 -S < %s |  FileCheck %s

; Check that we won't interleave by more than "best known" estimated trip count.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = dso_local global [5 x i32] zeroinitializer, align 16
@b = dso_local global [5 x i32] zeroinitializer, align 16

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @_Z3fooi(i32 %M) local_unnamed_addr {
; CHECK-LABEL: @_Z3fooi(
; CHECK:       [[VECTOR_BODY:vector\.body]]:
; CHECK:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
;
entry:
  %cmp8 = icmp sgt i32 %M, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %M to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [5 x i32], ptr @b, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %1 = trunc i64 %indvars.iv to i32
  %mul = mul nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds [5 x i32], ptr @a, i64 0, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %2, %mul
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !prof !1
}

!1 = !{!"branch_weights", i32 1, i32 9}
