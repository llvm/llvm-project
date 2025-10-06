; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

; Test PHIs that are uniform because they have a common/constant value over
; the divergent paths.

; Loop is uniform because loop exit PHI has constant value over all internal
; divergent paths.
define amdgpu_kernel void @no_divergent_exit1(i32 %a, i32 %b, i32 %c) #0 {
; CHECK-LABEL: for function 'no_divergent_exit1'
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
; CHECK: DIVERGENT: %div.cond =
  br label %header

header:
  %loop.b = phi i32 [ %b, %entry ], [ %new.b, %body.1 ], [ %new.b, %body.2 ]
; CHECK-NOT: DIVERGENT: %loop.b =
  %loop.c = phi i32 [ %c, %entry ], [ %loop.c, %body.1 ], [ %new.c, %body.2 ]
; CHECK: DIVERGENT: %loop.c =
  %exit.val = phi i32 [ %a, %entry ], [ %next.exit.val, %body.1 ], [ %next.exit.val, %body.2 ]
; CHECK-NOT: DIVERGENT: %exit.val =
  %exit.cond = icmp slt i32 %exit.val, 42
; CHECK-NOT: DIVERGENT: %exit.cond =
  br i1 %exit.cond, label %end, label %body.1
; CHECK-NOT: DIVERGENT: br i1 %exit.cond,

body.1:
  %new.b = add i32 %loop.b, 1
; CHECK-NOT: DIVERGENT: %new.b =
  %next.exit.val = add i32 %exit.val, 1
; CHECK-NOT: DIVERGENT: %next.exit.val =
  br i1 %div.cond, label %body.2, label %header
; CHECK: DIVERGENT: br i1 %div.cond,

body.2:
  %new.c = add i32 %loop.c, 1
; CHECK: DIVERGENT: %new.c =
  br label %header

end:
  ret void
}

; As no_divergent_exit1 but with merge block before exit.
define amdgpu_kernel void @no_divergent_exit2(i32 %a, i32 %b, i32 %c) #0 {
; CHECK-LABEL: for function 'no_divergent_exit2'
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
; CHECK: DIVERGENT: %div.cond =
  br label %header

header:
  %loop.b = phi i32 [ %b, %entry ], [ %merge.b, %merge ]
; CHECK-NOT: DIVERGENT: %loop.b =
  %loop.c = phi i32 [ %c, %entry ], [ %merge.c, %merge ]
; CHECK: DIVERGENT: %loop.c =
  %exit.val = phi i32 [ %a, %entry ], [ %next.exit.val, %merge ]
; CHECK-NOT: DIVERGENT: %exit.val =
  %exit.cond = icmp slt i32 %exit.val, 42
; CHECK-NOT: DIVERGENT: %exit.cond =
  br i1 %exit.cond, label %end, label %body.1
; CHECK-NOT: DIVERGENT: br i1 %exit.cond,

body.1:
  %new.b = add i32 %loop.b, 1
; CHECK-NOT: DIVERGENT: %new.b =
  %next.exit.val = add i32 %exit.val, 1
; CHECK-NOT: DIVERGENT: %next.exit.val =
  br i1 %div.cond, label %body.2, label %merge
; CHECK: DIVERGENT: br i1 %div.cond,

body.2:
  %new.c = add i32 %loop.c, 1
; CHECK: DIVERGENT: %new.c =
  br label %merge

merge:
  %merge.b = phi i32 [ %new.b, %body.1 ], [ %new.b, %body.2 ]
; CHECK-NOT: DIVERGENT: %merge.b =
  %merge.c = phi i32 [ %loop.c, %body.1 ], [ %new.c, %body.2 ]
; CHECK: DIVERGENT: %merge.c =
  br label %header

end:
  ret void
}

; Test PHI with constant value over divergent path without a loop.
define amdgpu_kernel void @no_loop_phi_divergence(i32 %a) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
; CHECK-NOT: DIVERGENT: %uni.cond =
  %div.cond = icmp slt i32 %tid, 0
; CHECK: DIVERGENT: %div.cond =
  br i1 %uni.cond, label %div.branch.block, label %merge
; CHECK-NOT: DIVERGENT: br i1 %uni.cond,

div.branch.block:
  br i1 %div.cond, label %div.block.1, label %div.block.2
; CHECK: DIVERGENT: br i1 %div.cond,

div.block.1:
  br label %merge

div.block.2:
  br label %merge

merge:
  %uni.val = phi i32 [ 0, %entry ], [ 1, %div.block.1 ], [ 1, %div.block.2 ]
; CHECK-NOT: DIVERGENT: %uni.val =
  %div.val = phi i32 [ 0, %entry ], [ 1, %div.block.1 ], [ 2, %div.block.2 ]
; CHECK: DIVERGENT: %div.val =
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
