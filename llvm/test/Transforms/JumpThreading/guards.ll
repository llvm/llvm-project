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
  call void(i1, ...) @llvm.experimental.guard( i1 %condGuard )[ "deopt"() ]
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
  call void(i1, ...) @llvm.experimental.guard( i1 %condGuard )[ "deopt"() ]
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
  call void(i1, ...) @llvm.experimental.guard( i1 %condGuard )[ "deopt"() ]
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
  call void(i1, ...) @llvm.experimental.guard( i1 %condGuard )[ "deopt"() ]
  ret i32 %retVal
}
