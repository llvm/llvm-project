; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

declare i32 @llvm.amdgcn.workitem.id.x()

; Divergent exit: the exit phi selects a different constant along each in-cycle
; exit edge and threads leave divergently.
define amdgpu_kernel void @divergent_cycle_exit_phi(i32 %n) {
; CHECK-LABEL: UniformityInfo for function 'divergent_cycle_exit_phi':
; CHECK: DIVERGENT:   %acc
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %body ]
  %exit.cond = icmp uge i32 %iv, %n
  br i1 %exit.cond, label %exit, label %body

body:
  %div.cond = icmp eq i32 %tid, 0
  %iv.next = add i32 %iv, 1
  br i1 %div.cond, label %exit, label %loop

exit:
  %acc = phi i32 [ 1, %body ], [ 0, %loop ]
  ret void
}

; Uniform: both in-cycle exit edges carry the same value.
define amdgpu_kernel void @uniform_cycle_exit_phi(i32 %n) {
; CHECK-LABEL: UniformityInfo for function 'uniform_cycle_exit_phi':
; CHECK-NOT: DIVERGENT:   %acc
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %body ]
  %exit.cond = icmp uge i32 %iv, %n
  br i1 %exit.cond, label %exit, label %body

body:
  %div.cond = icmp eq i32 %tid, 0
  %iv.next = add i32 %iv, 1
  br i1 %div.cond, label %exit, label %loop

exit:
  %acc = phi i32 [ 7, %body ], [ 7, %loop ]
  ret void
}

; Nested cycles: the divergent branch is in the inner cycle but the exit edge
; leaves both cycles at once.
define amdgpu_kernel void @nested_cycle_exit_phi() {
; CHECK-LABEL: UniformityInfo for function 'nested_cycle_exit_phi':
; CHECK: DIVERGENT:   %acc
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %outer.header

outer.header:
  %o = phi i32 [ 0, %entry ], [ %o.next, %outer.latch ]
  %o.cond = icmp slt i32 %o, 2
  br i1 %o.cond, label %inner.header, label %exit

inner.header:
  %i = phi i32 [ 0, %outer.header ], [ %i.next, %inner.body ]
  %i.cond = icmp slt i32 %i, 2
  br i1 %i.cond, label %inner.body, label %outer.latch

inner.body:
  %i.next = add i32 %i, 1
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %inner.header

outer.latch:
  %o.next = add i32 %o, 1
  br label %outer.header

exit:
  %acc = phi i32 [ 1, %inner.body ], [ 0, %outer.header ]
  ret void
}

; Uniform multi-predecessor exit reached only via uniform branches, alongside a
; separate divergent exit from the same cycle.
define amdgpu_kernel void @uniform_multi_exit_cycle_phi(i32 %n) {
; CHECK-LABEL: UniformityInfo for function 'uniform_multi_exit_cycle_phi':
; CHECK-NOT: DIVERGENT:   %acc
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %uni.cond = icmp slt i32 %n, 3
  br i1 %uni.cond, label %body, label %exit.uniform

body:
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit.divergent, label %mid

mid:
  %uni.cond2 = icmp sgt i32 %n, 1
  br i1 %uni.cond2, label %exit.uniform, label %loop

exit.divergent:
  ret void

exit.uniform:
  %acc = phi i32 [ 0, %loop ], [ 1, %mid ]
  ret void
}
