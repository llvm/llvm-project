; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

; CHECK=LABEL: UniformityInfo for function 'basic':
; CHECK: CYCLES ASSSUMED DIVERGENT:
; CHECK:   depth=1: entries(P T) Q
define amdgpu_kernel void @basic(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 br i1 %cond.div, label %T, label %P

P:
; CHECK:   DIVERGENT:   %pp.phi =
; CHECK: DIVERGENT:      %pp =
  %pp.phi  = phi i32 [ %a, %entry], [ %b, %T ]
  %pp = add i32 %b, 1
  br label %Q

Q:
; CHECK: DIVERGENT:   %qq =
; CHECK:   DIVERGENT:   %qq.div =
  %qq = add i32 %b, 1
  %qq.div = add i32 %pp.phi, 1
  br i1 %cond.uni, label %T, label %exit

T:
; CHECK:   DIVERGENT:   %t.phi =
; CHECK: DIVERGENT:     %tt =
  %t.phi = phi i32 [ %qq, %Q ], [ %a, %entry ]
  %tt = add i32 %b, 1
  br label %P

exit:
; CHECK-NOT: DIVERGENT:     %ee =
  %ee = add i32 %b, 1
  ret void
}

; CHECK=LABEL: UniformityInfo for function 'nested':
; CHECK: CYCLES ASSSUMED DIVERGENT:
; CHECK:  depth=1: entries(P T) Q A C B
define amdgpu_kernel void @nested(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 br i1 %cond.div, label %T, label %P

P:
 %pp.phi  = phi i32 [ %a, %entry], [ %b, %T ]
 %pp = add i32 %b, 1
 br i1 %cond.uni, label %B, label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.uni, label %T, label %exit

A:
  %aa = add i32 %b, 1
  br label %B

B:
  %bb = add i32 %b, 1
  br label %C

C:
  %cc = add i32 %b, 1
  br i1 %cond.uni, label %Q, label %A

T:
  %t.phi = phi i32 [ %qq, %Q ], [ %a, %entry ]
  %tt = add i32 %b, 1
  br i1 %cond.uni, label %A, label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
