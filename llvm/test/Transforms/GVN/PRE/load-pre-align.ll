; RUN: opt < %s -passes=gvn -S | FileCheck %s
; RUN: opt < %s -passes="gvn<load-pre>" -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"

@p = external global i32

define i32 @test(i32 %n) nounwind {
; CHECK-LABEL: @test(
entry:
  br label %for.cond

; loads aligned greater than the memory should not be moved past conditionals
; CHECK-NOT: load
; CHECK: br i1

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %indvar.next, %for.inc ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:
; ...but PRE can still move the load out of for.end to here.
; CHECK: for.cond.for.end_crit_edge:
; CHECK-NEXT: load
  br label %for.end

for.body:
  %tmp3 = load i32, ptr @p, align 8
  %dec = add i32 %tmp3, -1
  store i32 %dec, ptr @p
  %cmp6 = icmp slt i32 %dec, 0
  br i1 %cmp6, label %for.body.for.end_crit_edge, label %for.inc

for.body.for.end_crit_edge:
  br label %for.end

for.inc:
  %indvar.next = add i32 %i.0, 1
  br label %for.cond

for.end:
  %tmp9 = load i32, ptr @p, align 8
  ret i32 %tmp9
}
