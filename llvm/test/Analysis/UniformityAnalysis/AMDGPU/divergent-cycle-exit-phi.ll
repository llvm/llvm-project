; RUN: opt -mtriple=amdgpu7.00-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

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

; The divergent branch is the header's terminator, and its paths leave the cycle
; through different exiting blocks. Threads leave in different iterations, so
; the exit phi is divergent although its incoming values are identical, and so
; is the use of %i outside the cycle.
define amdgpu_kernel void @header_divergent_cycle_exit_phi(i32 %n, i32 %m) {
; CHECK-LABEL: UniformityInfo for function 'header_divergent_cycle_exit_phi':
; CHECK: DIVERGENT:   %i.lcssa
; CHECK: DIVERGENT:   %r
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp ult i32 %tid, 7
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  br i1 %div.cond, label %early, label %latch

early:
  %stop = icmp eq i32 %i, %m
  br i1 %stop, label %exit, label %latch

latch:
  %i.next = add i32 %i, 1
  %exit.cond = icmp eq i32 %i.next, %n
  br i1 %exit.cond, label %exit, label %loop

exit:
  %i.lcssa = phi i32 [ %i, %early ], [ %i, %latch ]
  %r = add i32 %i, 1000
  ret void
}

; The divergent branch is the header's terminator, but its paths join inside the
; cycle, and all threads leave the cycle in the same iteration.
define amdgpu_kernel void @header_divergent_uniform_cycle_exit_phi(i32 %n) {
; CHECK-LABEL: UniformityInfo for function 'header_divergent_uniform_cycle_exit_phi':
; CHECK-NOT: DIVERGENT:   %i.lcssa
; CHECK-NOT: DIVERGENT:   %r
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp ult i32 %tid, 7
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  br i1 %div.cond, label %then, label %latch

then:
  br label %latch

latch:
  %i.next = add i32 %i, 1
  %exit.cond = icmp eq i32 %i.next, %n
  br i1 %exit.cond, label %exit, label %loop

exit:
  %i.lcssa = phi i32 [ %i, %latch ]
  %r = add i32 %i, 1000
  ret void
}

; Nested cycles: the divergent branch is the inner header's terminator, and its
; paths leave the inner cycle, but not the outer one, through different exiting
; blocks.
define amdgpu_kernel void @nested_header_divergent_cycle_exit_phi(i32 %n, i32 %m) {
; CHECK-LABEL: UniformityInfo for function 'nested_header_divergent_cycle_exit_phi':
; CHECK: DIVERGENT:   %i.lcssa
; CHECK-NOT: DIVERGENT:   %o.next
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp ult i32 %tid, 7
  br label %outer.header

outer.header:
  %o = phi i32 [ 0, %entry ], [ %o.next, %outer.latch ]
  br label %inner.header

inner.header:
  %i = phi i32 [ 0, %outer.header ], [ %i.next, %inner.latch ]
  br i1 %div.cond, label %inner.early, label %inner.latch

inner.early:
  %stop = icmp eq i32 %i, %m
  br i1 %stop, label %outer.latch, label %inner.latch

inner.latch:
  %i.next = add i32 %i, 1
  %inner.cond = icmp eq i32 %i.next, %n
  br i1 %inner.cond, label %outer.latch, label %inner.header

outer.latch:
  %i.lcssa = phi i32 [ %i, %inner.early ], [ %i, %inner.latch ]
  %o.next = add i32 %o, 1
  %outer.cond = icmp eq i32 %o.next, 4
  br i1 %outer.cond, label %exit, label %outer.header

exit:
  ret void
}
