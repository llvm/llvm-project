; RUN: opt < %s -passes=loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -S | FileCheck --check-prefixes=CHECK,PREDICATED %s
; RUN: opt < %s -passes=loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -prefer-predicate-over-epilogue=scalar-epilogue -S | FileCheck --check-prefixes=CHECK,SCALAR %s

; This file contains the same function but with different trip-count PGO hints

; The function is vectorized if there are no trip-count hints
define i32 @foo_no_trip_count(ptr %a, ptr %b, ptr %c, i32 %bound) {
; CHECK-LABEL: @foo_no_trip_count(
; PREDICATED: vector.body
; SCALAR: vector.body
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %idx = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %a.index = getelementptr inbounds [32 x i8], ptr %a, i32 0, i32 %idx
  %0 = load i8, ptr %a.index, align 1
  %b.index = getelementptr inbounds [32 x i8], ptr %b, i32 0, i32 %idx
  %1 = load i8, ptr %b.index, align 1
  %2 = add i8 %0, %1
  %c.index = getelementptr inbounds [32 x i8], ptr %c, i32 0, i32 %idx
  store i8 %2, ptr %c.index, align 1
  %inc = add nsw i32 %idx, 1
  %exitcond = icmp eq i32 %idx, %bound
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

; If trip-count is equal to 4, the function is not vectorised
define i32 @foo_low_trip_count(ptr %a, ptr %b, ptr %c, i32 %bound) {
; CHECK-LABEL: @foo_low_trip_count(
; PREDICATED-NOT: vector.body
; SCALAR-NOT: vector.body
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %idx = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %a.index = getelementptr inbounds [32 x i8], ptr %a, i32 0, i32 %idx
  %0 = load i8, ptr %a.index, align 1
  %b.index = getelementptr inbounds [32 x i8], ptr %b, i32 0, i32 %idx
  %1 = load i8, ptr %b.index, align 1
  %2 = add i8 %0, %1
  %c.index = getelementptr inbounds [32 x i8], ptr %c, i32 0, i32 %idx
  store i8 %2, ptr %c.index, align 1
  %inc = add nsw i32 %idx, 1
  %exitcond = icmp eq i32 %idx, %bound
  br i1 %exitcond, label %for.end, label %for.body, !prof !0

for.end:                                          ; preds = %for.body
  ret i32 0
}

; If trip-count is equal to 10, the function is vectorised when predicated tail folding is chosen
define i32 @foo_mid_trip_count(ptr %a, ptr %b, ptr %c, i32 %bound) {
; CHECK-LABEL: @foo_mid_trip_count(
; PREDICATED: vector.body
; SCALAR-NOT: vector.body
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %idx = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %a.index = getelementptr inbounds [32 x i8], ptr %a, i32 0, i32 %idx
  %0 = load i8, ptr %a.index, align 1
  %b.index = getelementptr inbounds [32 x i8], ptr %b, i32 0, i32 %idx
  %1 = load i8, ptr %b.index, align 1
  %2 = add i8 %0, %1
  %c.index = getelementptr inbounds [32 x i8], ptr %c, i32 0, i32 %idx
  store i8 %2, ptr %c.index, align 1
  %inc = add nsw i32 %idx, 1
  %exitcond = icmp eq i32 %idx, %bound
  br i1 %exitcond, label %for.end, label %for.body, !prof !1

for.end:                                          ; preds = %for.body
  ret i32 0
}

; If trip-count is equal to 40, the function is always vectorised
define i32 @foo_high_trip_count(ptr %a, ptr %b, ptr %c, i32 %bound) {
; CHECK-LABEL: @foo_high_trip_count(
; PREDICATED: vector.body
; SCALAR: vector.body
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %idx = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %a.index = getelementptr inbounds [32 x i8], ptr %a, i32 0, i32 %idx
  %0 = load i8, ptr %a.index, align 1
  %b.index = getelementptr inbounds [32 x i8], ptr %b, i32 0, i32 %idx
  %1 = load i8, ptr %b.index, align 1
  %2 = add i8 %0, %1
  %c.index = getelementptr inbounds [32 x i8], ptr %c, i32 0, i32 %idx
  store i8 %2, ptr %c.index, align 1
  %inc = add nsw i32 %idx, 1
  %exitcond = icmp eq i32 %idx, %bound
  br i1 %exitcond, label %for.end, label %for.body, !prof !2

for.end:                                          ; preds = %for.body
  ret i32 0
}

!0 = !{!"branch_weights", i32 10, i32 30}
!1 = !{!"branch_weights", i32 10, i32 90}
!2 = !{!"branch_weights", i32 10, i32 390}
