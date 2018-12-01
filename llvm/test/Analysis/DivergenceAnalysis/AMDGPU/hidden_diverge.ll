; RUN: opt -mtriple amdgcn-unknown-amdhsa -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s

define amdgpu_kernel void @hidden_diverge(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: Printing analysis 'Legacy Divergence Analysis' for function 'hidden_diverge'
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond.var = icmp slt i32 %tid, 0
  br i1 %cond.var, label %B, label %C ; divergent
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

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
