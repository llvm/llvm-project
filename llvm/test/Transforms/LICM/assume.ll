; RUN: opt -licm -basicaa < %s -S | FileCheck %s

define void @f_0(i1 %p) nounwind ssp {
; CHECK-LABEL: @f_0(
entry:
  br label %for.body

for.body:
  br i1 undef, label %if.then, label %for.cond.backedge

for.cond.backedge:
  br i1 undef, label %for.end104, label %for.body

if.then:
  br i1 undef, label %if.then27, label %if.end.if.end.split_crit_edge.critedge

if.then27:
; CHECK: tail call void @llvm.assume
  tail call void @llvm.assume(i1 %p)
  br label %for.body61.us

if.end.if.end.split_crit_edge.critedge:
  br label %for.body61

for.body61.us:
  br i1 undef, label %for.cond.backedge, label %for.body61.us

for.body61:
  br i1 undef, label %for.cond.backedge, label %for.body61

for.end104:
  ret void
}

define void @f_1(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @f_1(
; CHECK: %val = load i32, i32* %ptr
; CHECK-NEXT:  br label %loop

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call void @llvm.assume(i1 %cond)
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

declare void @llvm.assume(i1)
