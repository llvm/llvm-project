; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; temporal-divergent use of value carried by divergent loop
define amdgpu_kernel void @temporal_diverge(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'temporal_diverge':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br label %H

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %H ]
  %uni.inc = add i32 %uni.merge.h, 1
; CHECK: DIVERGENT: %div.exitx =
  %div.exitx = icmp slt i32 %tid, 0
; CHECK: DIVERGENT: br i1 %div.exitx,
  br i1 %div.exitx, label %X, label %H ; divergent branch

X:
; CHECK: DIVERGENT:  %div.user =
  %div.user = add i32 %uni.inc, 5
  ret void
}

define amdgpu_kernel void @phi_at_exit(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'phi_at_exit':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %H, label %X

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %H ]
  %uni.inc = add i32 %uni.merge.h, 1
; CHECK: DIVERGENT: %div.exitx =
  %div.exitx = icmp slt i32 %tid, 0
; CHECK: DIVERGENT: br i1 %div.exitx,
  br i1 %div.exitx, label %X, label %H ; divergent branch

X:
; CHECK: DIVERGENT:  %div.phi =
  %div.phi = phi i32 [ 0, %entry], [ %uni.inc, %H ]
  %div.user = add i32 %div.phi, 5
  ret void
}

define amdgpu_kernel void @phi_after_exit(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'phi_after_exit':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %H, label %Y

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %H ]
  %uni.inc = add i32 %uni.merge.h, 1
; CHECK: DIVERGENT: %div.exitx =
  %div.exitx = icmp slt i32 %tid, 0
; CHECK: DIVERGENT: br i1 %div.exitx,
  br i1 %div.exitx, label %X, label %H ; divergent branch

X:
  br label %Y

Y:
; CHECK: DIVERGENT:  %div.phi =
  %div.phi = phi i32 [ 0, %entry], [ %uni.inc, %X ]
  %div.user = add i32 %div.phi, 5
  ret void
}

; temporal-divergent use of value carried by divergent loop inside a top-level loop
define amdgpu_kernel void @temporal_diverge_inloop(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'temporal_diverge_inloop':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br label %G

G:
  br label %H

H:
  %uni.merge.h = phi i32 [ 0, %G ], [ %uni.inc, %H ]
  %uni.inc = add i32 %uni.merge.h, 1
  %div.exitx = icmp slt i32 %tid, 0
  br i1 %div.exitx, label %X, label %H ; divergent branch
; CHECK: DIVERGENT: %div.exitx =
; CHECK: DIVERGENT: br i1 %div.exitx,

X:
; CHECK: DIVERGENT: %div.user =
  %div.user = add i32 %uni.inc, 5
  br i1 %uni.cond, label %G, label %Y

Y:
; CHECK: DIVERGENT: %div.alsouser =
  %div.alsouser = add i32 %uni.inc, 5
  ret void
}


; temporal-uniform use of a value, definition and users are carried by a
; surrounding divergent loop
define amdgpu_kernel void @temporal_uniform_indivloop(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'temporal_uniform_indivloop':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br label %G

G:
  br label %H

H:
  %uni.merge.h = phi i32 [ 0, %G ], [ %uni.inc, %H ]
  %uni.inc = add i32 %uni.merge.h, 1
  br i1 %uni.cond, label %X, label %H

X:
  %uni.user = add i32 %uni.inc, 5
  %div.exity = icmp slt i32 %tid, 0
; CHECK: DIVERGENT: %div.exity =
  br i1 %div.exity, label %G, label %Y
; CHECK: DIVERGENT: br i1 %div.exity,

Y:
  %div.alsouser = add i32 %uni.inc, 5
  ret void
; CHECK: DIVERGENT: %div.alsouser =
}


; temporal-divergent use of value carried by divergent loop, user is inside sibling loop
define amdgpu_kernel void @temporal_diverge_loopuser(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'temporal_diverge_loopuser':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br label %H

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %H ]
  %uni.inc = add i32 %uni.merge.h, 1
  %div.exitx = icmp slt i32 %tid, 0
  br i1 %div.exitx, label %X, label %H ; divergent branch
; CHECK: DIVERGENT: %div.exitx =
; CHECK: DIVERGENT: br i1 %div.exitx,

X:
  br label %G

G:
; C HECK: DIVERGENT: %div.user =
  %div.user = add i32 %uni.inc, 5
  br i1 %uni.cond, label %G, label %Y
; CHECK: DIVERGENT: %div.user =

Y:
  ret void
}

; temporal-divergent use of value carried by divergent loop, user is inside
; sibling loop, defs and use are carried by a uniform loop
define amdgpu_kernel void @temporal_diverge_loopuser_nested(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: for function 'temporal_diverge_loopuser_nested':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br label %G

G:
  br label %H

H:
  %uni.merge.h = phi i32 [ 0, %G ], [ %uni.inc, %H ]
  %uni.inc = add i32 %uni.merge.h, 1
  %div.exitx = icmp slt i32 %tid, 0
  br i1 %div.exitx, label %X, label %H ; divergent branch
; CHECK: DIVERGENT: %div.exitx =
; CHECK: DIVERGENT: br i1 %div.exitx,

X:
; CHECK: DIVERGENT: %div.user =
  %div.user = add i32 %uni.inc, 5
  br i1 %uni.cond, label %X, label %G

Y:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
