; RUN: opt -mtriple amdgpu7.00-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

define amdgpu_kernel void @exit_into_next_loop(i32 %n, i1 %c) {
; CHECK-LABEL: UniformityInfo for function 'exit_into_next_loop':
; CHECK: CYCLES WITH DIVERGENT EXIT:
; CHECK-NEXT: depth=1: entries(outer) inner body
; CHECK: DIVERGENT: br i1 %div.cond, label %next, label %inner
; CHECK: DIVERGENT: %outer.use = add i32 %o, 0
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %outer

outer:
  %o = phi i32 [ 0, %entry ], [ %o.next, %inner ]
  %o.next = add i32 %o, 1
  br label %inner

inner:
  %i = phi i32 [ 0, %outer ], [ %i.next, %body ]
  %uni.cond = icmp slt i32 %i, %n
  br i1 %uni.cond, label %body, label %outer

body:
  %i.next = add i32 %i, 1
  %div.cond = icmp eq i32 %o, %tid
  br i1 %div.cond, label %next, label %inner

next:
  %outer.use = add i32 %o, 0
  br i1 %c, label %next, label %exit

exit:
  ret void
}

define amdgpu_kernel void @exit_into_inner_loop_of_next_nest(i32 %n, i1 %c1, i1 %c2, i1 %c3) {
; CHECK-LABEL: UniformityInfo for function 'exit_into_inner_loop_of_next_nest':
; CHECK: CYCLES WITH DIVERGENT EXIT:
; CHECK-NEXT: depth=1: entries(outer) inner body
; CHECK: DIVERGENT: br i1 %div.cond, label %next.inner, label %inner
; CHECK: DIVERGENT: %outer.use = add i32 %o, 0
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %outer

outer:
  %o = phi i32 [ 0, %entry ], [ %o.next, %inner ]
  %o.next = add i32 %o, 1
  br i1 %c1, label %inner, label %next.outer

inner:
  %i = phi i32 [ 0, %outer ], [ %i.next, %body ]
  %uni.cond = icmp slt i32 %i, %n
  br i1 %uni.cond, label %body, label %outer

body:
  %i.next = add i32 %i, 1
  %div.cond = icmp eq i32 %o, %tid
  br i1 %div.cond, label %next.inner, label %inner

next.outer:
  br i1 %c2, label %next.inner, label %exit

next.inner:
  br i1 %c3, label %next.inner, label %next.outer

exit:
  %outer.use = add i32 %o, 0
  ret void
}

define amdgpu_kernel void @uniform_exit_use_after_exit_into_next_loop(i32 %n, i1 %c) {
; CHECK-LABEL: UniformityInfo for function 'uniform_exit_use_after_exit_into_next_loop':
; CHECK: DIVERGENT: br i1 %div.cond, label %next, label %inner
; CHECK-NOT: DIVERGENT: %i.lcssa = phi i32 [ %i, %inner ]
; CHECK-NOT: DIVERGENT: %latch.use = add i32 %i, 1
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %outer

outer:
  %o = phi i32 [ 0, %entry ], [ %o.next, %latch ]
  %o.next = add i32 %o, 1
  br label %inner

inner:
  %i = phi i32 [ 0, %outer ], [ %i.next, %body ]
  %uni.cond = icmp slt i32 %i, %n
  br i1 %uni.cond, label %body, label %latch

body:
  %i.next = add i32 %i, 1
  %div.cond = icmp eq i32 %o, %tid
  br i1 %div.cond, label %next, label %inner

latch:
  %i.lcssa = phi i32 [ %i, %inner ]
  %latch.use = add i32 %i, 1
  br label %outer

next:
  br i1 %c, label %next, label %exit

exit:
  ret void
}
