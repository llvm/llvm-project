; RUN: opt -S -passes=indvars < %s | FileCheck %s

declare void @side_effect(i1)

define void @latch_dominating_0(i8 %start) {
; CHECK-LABEL: latch_dominating_0
 entry:
  %e = icmp slt i8 %start, 42
  br i1 %e, label %loop, label %exit

 loop:
; CHECK-LABEL: loop
  %idx = phi i8 [ %start, %entry ], [ %idx.inc, %be ]
  %idx.inc = add i8 %idx, 1
  %folds.to.true = icmp slt i8 %idx, 42
; CHECK: call void @side_effect(i1 true)
  call void @side_effect(i1 %folds.to.true)
  %c0 = icmp slt i8 %idx.inc, 42
  br i1 %c0, label %be, label %exit

 be:
; CHECK: call void @side_effect(i1 true)
  call void @side_effect(i1 %folds.to.true)
  %c1 = icmp slt i8 %idx.inc, 100
  br i1 %c1, label %loop, label %exit

 exit:
  ret void
}

define void @latch_dominating_1(i8 %start) {
; CHECK-LABEL: latch_dominating_1
 entry:
  %e = icmp slt i8 %start, 42
  br i1 %e, label %loop, label %exit

 loop:
; CHECK-LABEL: loop
  %idx = phi i8 [ %start, %entry ], [ %idx.inc, %be ]
  %idx.inc = add i8 %idx, 1
  %does.not.fold.to.true = icmp slt i8 %idx, 42
; CHECK: call void @side_effect(i1 %does.not.fold.to.true)
  call void @side_effect(i1 %does.not.fold.to.true)
  %c0 = icmp slt i8 %idx.inc, 42
  br i1 %c0, label %be, label %be

 be:
; CHECK: call void @side_effect(i1 %does.not.fold.to.true)
  call void @side_effect(i1 %does.not.fold.to.true)
  %c1 = icmp slt i8 %idx.inc, 100
  br i1 %c1, label %loop, label %exit

 exit:
  ret void
}
