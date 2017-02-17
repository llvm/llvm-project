; RUN: opt < %s -jump-threading -dce -S | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

declare i32 @f1()
declare i32 @f2()

define i32 @branch_implies_guard(i32 %a) {
; CHECK-LABEL: @branch_implies_guard(
  %cond = icmp slt i32 %a, 10
  br i1 %cond, label %T1, label %F1

T1:
; CHECK:       T1.split
; CHECK:         %v1 = call i32 @f1()
; CHECK-NEXT:    %retVal
; CHECK-NEXT:    br label %Merge
  %v1 = call i32 @f1()
  br label %Merge

F1:
; CHECK:       F1.split
; CHECK:         %v2 = call i32 @f2()
; CHECK-NEXT:    %retVal
; CHECK-NEXT:    %condGuard
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %condGuard
; CHECK-NEXT:    br label %Merge
  %v2 = call i32 @f2()
  br label %Merge

Merge:
; CHECK:       Merge
; CHECK-NOT:     call void(i1, ...) @llvm.experimental.guard(
  %retPhi = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %retVal = add i32 %retPhi, 10
  %condGuard = icmp slt i32 %a, 20
  call void(i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  ret i32 %retVal
}

define i32 @not_branch_implies_guard(i32 %a) {
; CHECK-LABEL: @not_branch_implies_guard(
  %cond = icmp slt i32 %a, 20
  br i1 %cond, label %T1, label %F1

T1:
; CHECK:       T1.split:
; CHECK-NEXT:    %v1 = call i32 @f1()
; CHECK-NEXT:    %retVal
; CHECK-NEXT:    %condGuard
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %condGuard
; CHECK-NEXT:    br label %Merge
  %v1 = call i32 @f1()
  br label %Merge

F1:
; CHECK:       F1.split:
; CHECK-NEXT:   %v2 = call i32 @f2()
; CHECK-NEXT:   %retVal
; CHECK-NEXT:   br label %Merge
  %v2 = call i32 @f2()
  br label %Merge

Merge:
; CHECK:       Merge
; CHECK-NOT:     call void(i1, ...) @llvm.experimental.guard(
  %retPhi = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %retVal = add i32 %retPhi, 10
  %condGuard = icmp sgt i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  ret i32 %retVal
}

define i32 @branch_overlaps_guard(i32 %a) {
; CHECK-LABEL: @branch_overlaps_guard(
  %cond = icmp slt i32 %a, 20
  br i1 %cond, label %T1, label %F1

T1:
; CHECK:        T1:
; CHECK-NEXT:      %v1 = call i32 @f1()
; CHECK-NEXT:      br label %Merge
  %v1 = call i32 @f1()
  br label %Merge

F1:
; CHECK:        F1:
; CHECK-NEXT:     %v2 = call i32 @f2()
; CHECK-NEXT:     br label %Merge
  %v2 = call i32 @f2()
  br label %Merge

Merge:
; CHECK:       Merge
; CHECK:         %condGuard = icmp slt i32 %a, 10
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  %retPhi = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %retVal = add i32 %retPhi, 10
  %condGuard = icmp slt i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  ret i32 %retVal
}

define i32 @branch_doesnt_overlap_guard(i32 %a) {
; CHECK-LABEL: @branch_doesnt_overlap_guard(
  %cond = icmp slt i32 %a, 10
  br i1 %cond, label %T1, label %F1

T1:
; CHECK:        T1:
; CHECK-NEXT:      %v1 = call i32 @f1()
; CHECK-NEXT:      br label %Merge
  %v1 = call i32 @f1()
  br label %Merge

F1:
; CHECK:        F1:
; CHECK-NEXT:     %v2 = call i32 @f2()
; CHECK-NEXT:     br label %Merge
  %v2 = call i32 @f2()
  br label %Merge

Merge:
; CHECK:       Merge
; CHECK:         %condGuard = icmp sgt i32 %a, 20
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  %retPhi = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %retVal = add i32 %retPhi, 10
  %condGuard = icmp sgt i32 %a, 20
  call void(i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  ret i32 %retVal
}

define i32 @not_a_diamond1(i32 %a, i1 %cond1) {
; CHECK-LABEL: @not_a_diamond1(
  br i1 %cond1, label %Pred, label %Exit

Pred:
; CHECK:       Pred:
; CHECK-NEXT:    switch i32 %a, label %Exit
  switch i32 %a, label %Exit [
    i32 10, label %Merge
    i32 20, label %Merge
  ]

Merge:
; CHECK:       Merge:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
; CHECK-NEXT:    br label %Exit
  call void(i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
  br label %Exit

Exit:
; CHECK:       Exit:
; CHECK-NEXT:    ret i32 %a
  ret i32 %a
}

define void @not_a_diamond2(i32 %a, i1 %cond1) {
; CHECK-LABEL: @not_a_diamond2(
  br label %Parent

Merge:
  call void(i1, ...) @llvm.experimental.guard(i1 %cond1)[ "deopt"() ]
  ret void

Pred:
; CHECK-NEXT:  Pred:
; CHECK-NEXT:    switch i32 %a, label %Exit
  switch i32 %a, label %Exit [
    i32 10, label %Merge
    i32 20, label %Merge
  ]

Parent:
  br label %Pred

Exit:
; CHECK:       Merge:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
; CHECK-NEXT:    ret void
  ret void
}
