; RUN: opt -S -passes=loop-unroll --debug-only=loop-unroll < %s 2>&1 | FileCheck %s -check-prefix=LOOP-UNROLL
; RUN: opt -S -passes='require<opt-remark-emit>,loop(loop-unroll-full)' --debug-only=loop-unroll < %s 2>&1 | FileCheck %s -check-prefix=LOOP-UNROLL-FULL

; REQUIRES: asserts

%struct.HIP_vector_type = type {  %union.anon }
%union.anon = type { <2 x float> }


; LOOP-UNROLL-LABEL: Loop Unroll: F[pragma_unroll] Loop %for.body
; LOOP-UNROLL-NEXT: Loop Size = 9
; LOOP-UNROLL-NEXT: runtime unrolling with count: 8
; LOOP-UNROLL-NEXT: Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; LOOP-UNROLL-NEXT: Trying runtime unrolling on Loop:
; LOOP-UNROLL-NEXT: Loop at depth 1 containing: %for.body<header><latch><exiting>
; LOOP-UNROLL-NEXT: Using epilog remainder.
; LOOP-UNROLL-NEXT: UNROLLING loop %for.body by 8 with run-time trip count!

; LOOP-UNROLL-FULL-LABEL: Loop Unroll: F[pragma_unroll] Loop %for.body
; LOOP-UNROLL-FULL-NEXT: Loop Size = 9
; LOOP-UNROLL-FULL-NEXT:  runtime unrolling with count: 8
; LOOP-UNROLL-FULL-NEXT: Not attempting partial/runtime unroll in FullLoopUnroll
define void @pragma_unroll(ptr %queue, i32 %num_elements) {
entry:
  %cmp5 = icmp sgt i32 %num_elements, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %add = add nuw nsw i32 %i.06, 1
  %idxprom = zext i32 %add to i64
  %arrayidx = getelementptr inbounds %struct.HIP_vector_type, ptr %queue, i64 %idxprom
  %idxprom1 = zext i32 %i.06 to i64
  %arrayidx2 = getelementptr inbounds %struct.HIP_vector_type, ptr %queue, i64 %idxprom1
  %0 = load i64, ptr %arrayidx, align 8
  store i64 %0, ptr %arrayidx2, align 8
  %exitcond = icmp ne i32 %add, %num_elements
  br i1 %exitcond, label %for.body, label %for.cond.cleanup.loopexit, !llvm.loop !1
}

; LOOP-UNROLL-LABEL: Loop Unroll: F[pragma_unroll_count1] Loop %for.body
; LOOP-UNROLL-NEXT: Loop Size = 9
; LOOP-UNROLL-NEXT: Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; LOOP-UNROLL-NEXT: Trying runtime unrolling on Loop:
; LOOP-UNROLL-NEXT: Loop at depth 1 containing: %for.body<header><latch><exiting>
; LOOP-UNROLL-NEXT: Using epilog remainder.
; LOOP-UNROLL-NEXT: UNROLLING loop %for.body by 5 with run-time trip count!

; LOOP-UNROLL-FULL-LABEL: Loop Unroll: F[pragma_unroll_count1] Loop %for.body
; LOOP-UNROLL-FULL-NEXT: Loop Size = 9
; LOOP-UNROLL-FULL-NEXT: Not attempting partial/runtime unroll in FullLoopUnroll
define void @pragma_unroll_count1(ptr %queue, i32 %num_elements) {
entry:
  %cmp5 = icmp sgt i32 %num_elements, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %add = add nuw nsw i32 %i.06, 1
  %idxprom = zext i32 %add to i64
  %arrayidx = getelementptr inbounds %struct.HIP_vector_type, ptr %queue, i64 %idxprom
  %idxprom1 = zext i32 %i.06 to i64
  %arrayidx2 = getelementptr inbounds %struct.HIP_vector_type, ptr %queue, i64 %idxprom1
  %0 = load i64, ptr %arrayidx, align 8
  store i64 %0, ptr %arrayidx2, align 8
  %exitcond = icmp ne i32 %add, %num_elements
  br i1 %exitcond, label %for.body, label %for.cond.cleanup.loopexit, !llvm.loop !3
}

; LOOP-UNROLL: llvm.loop.unroll.disable
; LOOP-UNROLL-FULL: llvm.loop.unroll.enable
!0 = !{!"llvm.loop.unroll.enable"}
!1 = distinct !{!1, !0}

!2 = !{!"llvm.loop.unroll.count", i32 5}
!3 = distinct !{!3, !2}
