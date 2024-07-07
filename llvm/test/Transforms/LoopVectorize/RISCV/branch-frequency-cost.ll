; RUN: opt -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -riscv-v-register-bit-width-lmul=1 -passes='require<profile-summary>,loop-vectorize' -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s

; Check that branch weights make a difference when computing cost of scalar loop

define void @foo_with_wts(ptr %A, ptr %B, i32 %n) {
; CHECK: LV: Checking a loop in 'foo_with_wts'
; CHECK: LV: Scalar loop costs: [[COST:[0-9]+]].
; CHECK-NOT: vector.body
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %0 = trunc i64 %indvars.iv to i32
  %rem = urem i32 %0, 100
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %for.inc, !prof !0

if.then:
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx3, align 4
  %udiv1 = udiv i32 %2, %1
  %udiv2 = udiv i32 %2, %1
  %udiv3 = udiv i32 %2, %1
  %udiv4 = udiv i32 %2, %1
  %udiv5 = udiv i32 %2, %1
  %udiv6 = udiv i32 %2, %1
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}


define void @foo_no_wts(ptr %A, ptr %B, i32 %n) {
; CHECK: LV: Checking a loop in 'foo_no_wts'
; CHECK-NOT: LV: Scalar loop costs: [[COST]].
; CHECK: vector.body
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %0 = trunc i64 %indvars.iv to i32
  %rem = urem i32 %0, 100
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx3, align 4
  %udiv1 = udiv i32 %2, %1
  %udiv2 = udiv i32 %2, %1
  %udiv3 = udiv i32 %2, %1
  %udiv4 = udiv i32 %2, %1
  %udiv5 = udiv i32 %2, %1
  %udiv6 = udiv i32 %2, %1
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}

!0 = !{!"branch_weights", i32 1, i32 100}

; Currently, the loop vectorizer only utilizes BranchFrequencyInfo in the
; presence of ProfileSummaryInfo (https://reviews.llvm.org/D144953)
; Fabricate a summary which won't be used:
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 100}
!8 = !{!"NumCounts", i64 200}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 1000, i32 1}
!13 = !{i32 990000, i64 300, i32 10}
!14 = !{i32 999999, i64 5, i32 100}
