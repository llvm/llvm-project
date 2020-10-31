; This test verifies that the loop vectorizer will not vectorizes low trip count
; loops that require runtime checks (Trip count is computed with profile info).
; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -loop-vectorize-with-block-frequency -S | FileCheck %s

target datalayout = "E-m:e-p:32:32-i64:32-f64:32:64-a:0:32-n32-S128"

@tab = common global [32 x i8] zeroinitializer, align 1

define i32 @foo_low_trip_count1(i32 %bound) {
; Simple loop with low tripcount. Should not be vectorized.

; CHECK-LABEL: @foo_low_trip_count1(
; CHECK-NOT: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, %bound
  br i1 %exitcond, label %for.end, label %for.body, !prof !1

for.end:                                          ; preds = %for.body
  ret i32 0
}

define i32 @foo_low_trip_count2(i32 %bound) !prof !0 {
; The loop has a same invocation count with the function, but has a low
; trip_count per invocation and not worth to vectorize.

; CHECK-LABEL: @foo_low_trip_count2(
; CHECK-NOT: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, %bound
  br i1 %exitcond, label %for.end, label %for.body, !prof !1

for.end:                                          ; preds = %for.body
  ret i32 0
}

define i32 @foo_low_trip_count3(i1 %cond, i32 %bound) !prof !0 {
; The loop has low invocation count compare to the function invocation count,
; but has a high trip count per invocation. Vectorize it.

; CHECK-LABEL: @foo_low_trip_count3(
; CHECK:  [[VECTOR_BODY:vector\.body]]:
; CHECK:    br i1 [[TMP9:%.*]], label [[MIDDLE_BLOCK:%.*]], label %[[VECTOR_BODY]], !prof [[LP3:\!.*]],
; CHECK:  [[FOR_BODY:for\.body]]:
; CHECK:    br i1 [[EXITCOND:%.*]], label [[FOR_END_LOOPEXIT:%.*]], label %[[FOR_BODY]], !prof [[LP6:\!.*]],
entry:
  br i1 %cond, label %for.preheader, label %for.end, !prof !2

for.preheader:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, %bound
  br i1 %exitcond, label %for.end, label %for.body, !prof !3

for.end:                                          ; preds = %for.body
  ret i32 0
}

define i32 @foo_low_trip_count_icmp_sgt(i32 %bound) {
; Simple loop with low tripcount and inequality test for exit.
; Should not be vectorized.

; CHECK-LABEL: @foo_low_trip_count_icmp_sgt(
; CHECK-NOT: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp sgt i32 %i.08, %bound
  br i1 %exitcond, label %for.end, label %for.body, !prof !1

for.end:                                          ; preds = %for.body
  ret i32 0
}

define i32 @const_low_trip_count() {
; Simple loop with constant, small trip count and no profiling info.

; CHECK-LABEL: @const_low_trip_count
; CHECK-NOT: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp slt i32 %i.08, 2
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret i32 0
}

define i32 @const_large_trip_count() {
; Simple loop with constant large trip count and no profiling info.

; CHECK-LABEL: @const_large_trip_count
; CHECK: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp slt i32 %i.08, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret i32 0
}

define i32 @const_small_trip_count_step() {
; Simple loop with static, small trip count and no profiling info.

; CHECK-LABEL: @const_small_trip_count_step
; CHECK-NOT: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 5
  %exitcond = icmp slt i32 %i.08, 10
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret i32 0
}

define i32 @const_trip_over_profile() {
; constant trip count takes precedence over profile data

; CHECK-LABEL: @const_trip_over_profile
; CHECK: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp slt i32 %i.08, 1000
  br i1 %exitcond, label %for.body, label %for.end, !prof !1

for.end:                                          ; preds = %for.body
  ret i32 0
}

; CHECK: [[LP3]] = !{!"branch_weights", i32 10, i32 2490}
; CHECK: [[LP6]] = !{!"branch_weights", i32 10, i32 0}
; original loop has latchExitWeight=10 and backedgeTakenWeight=10,000,
; therefore estimatedBackedgeTakenCount=1,000 and estimatedTripCount=1,001.
; Vectorizing by 4 produces estimatedTripCounts of 1,001/4=250 and 1,001%4=1
; for vectorized and remainder loops, respectively, therefore their
; estimatedBackedgeTakenCounts are 249 and 0, and so the weights recorded with
; loop invocation weights of 10 are the above {10, 2490} and {10, 0}.

!0 = !{!"function_entry_count", i64 100}
!1 = !{!"branch_weights", i32 100, i32 0}
!2 = !{!"branch_weights", i32 10, i32 90}
!3 = !{!"branch_weights", i32 10, i32 10000}
