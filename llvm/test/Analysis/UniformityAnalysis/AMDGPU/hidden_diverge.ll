; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

define amdgpu_kernel void @hidden_diverge(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'hidden_diverge'
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond.var = icmp slt i32 %tid, 0
  br i1 %cond.var, label %B, label %C ; divergent
; CHECK:  DIVERGENT: %cond.var =
; CHECK: DIVERGENT: br i1 %cond.var,
B:
  %cond.uni = icmp slt i32 %n, 0
  br i1 %cond.uni, label %C, label %merge ; uniform
; CHECK-NOT: DIVERGENT: br i1 %cond.uni,
C:
  %phi.var.hidden = phi i32 [ 1, %entry ], [ 2, %B  ]
; CHECK: DIVERGENT: %phi.var.hidden = phi i32
  br label %merge
merge:
  %phi.ipd = phi i32 [ %a, %B ], [ %b, %C ]
; CHECK: DIVERGENT: %phi.ipd = phi i32
  ret void
}

define amdgpu_kernel void @hidden_loop_ipd(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'hidden_loop_ipd'
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond.var = icmp slt i32 %tid, 0
; CHECK: DIVERGENT: %cond.var = icmp
  %cond.uni = icmp slt i32 %n, 0
; CHECK-NOT: DIVERGENT: %cond.uni = icmp
  br label %for.header
for.header:
  br i1 %cond.var, label %A, label %B
A:
  br label %C
B:
  br label %C
C:
  br i1 %cond.uni, label %E, label %D
D:
  br i1 %cond.var, label %for.header, label %F

E:
  %e.lcssa.uni = phi i32 [ 0, %C ]
; CHECK-NOT: DIVERGENT: %e.lcssa.uni = phi i32
  br label %G

F:
  %f.lcssa.uni = phi i32 [ 1, %D ]
; CHECK-NOT: DIVERGENT: %f.lcssa.uni = phi i32
  br label %G

G:
  %g.join.var = phi i32 [ %e.lcssa.uni, %E ], [ %f.lcssa.uni, %F ]
; CHECK: DIVERGENT: %g.join.var = phi i32
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
