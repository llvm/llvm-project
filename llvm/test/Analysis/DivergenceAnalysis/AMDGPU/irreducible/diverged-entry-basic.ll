; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

define amdgpu_kernel void @divergent_cycle_1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: UniformityInfo for function 'divergent_cycle_1':
; CHECK: CYCLES ASSSUMED DIVERGENT:
; CHECK:   depth=1: entries(R P) S Q
; CHECK: CYCLES WITH DIVERGENT EXIT:
; CHECK:   depth=2: entries(S P) Q
; CHECK:   depth=1: entries(R P) S Q
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 br i1 %cond.uni, label %P, label %R

P:
; CHECK:   DIVERGENT:   %pp.phi =
  %pp.phi  = phi i32 [ %a, %entry], [ %b, %S ]
  %pp = add i32 %b, 1
  br label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.div, label %S, label %R

R:
  %rr = add i32 %b, 1
  br label %S

S:
; CHECK:   DIVERGENT:   %s.phi =
  %s.phi = phi i32 [ %qq, %Q ], [ %rr, %R ]
  %ss = add i32 %b, 1
  br i1 %cond.uni, label %exit, label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

define amdgpu_kernel void @uniform_cycle_1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: UniformityInfo for function 'uniform_cycle_1':
; CHECK-NOT: CYCLES ASSSUMED DIVERGENT:
; CHECK-NOT: CYCLES WITH DIVERGENT EXIT:
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 br i1 %cond.uni, label %P, label %T

P:
; CHECK-NOT:   DIVERGENT:   %pp.phi = phi i32
  %pp.phi  = phi i32 [ %a, %entry], [ %b, %T ]
  %pp = add i32 %b, 1
  br label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.div, label %S, label %R

R:
  %rr = add i32 %b, 1
  br label %S

S:
; CHECK:   DIVERGENT:   %s.phi =
  %s.phi = phi i32 [ %qq, %Q ], [ %rr, %R ]
  %ss = add i32 %b, 1
  br i1 %cond.uni, label %exit, label %T

T:
  %tt = add i32 %b, 1
  br label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
