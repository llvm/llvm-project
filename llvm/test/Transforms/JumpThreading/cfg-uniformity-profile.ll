; RUN: opt -S -passes=jump-threading -verify-each < %s | FileCheck %s

; A reconverged block may have a full-wave profile even when either predecessor
; executes with a partial mask. Drop both hints on the duplicate and the original.
; CHECK-LABEL: define void @clone_phi(
; CHECK: br i1 %x, label %yes, label %no, !prof ![[W:[0-9]+]]{{$}}
; CHECK: br i1 %y, label %yes, label %no, !prof ![[W]]{{$}}
define void @clone_phi(i1 %dispatch, i1 %x, i1 %y) !uniformity.profile !0 {
entry:
  br i1 %dispatch, label %left, label %right
left:
  call void @left_call()
  br label %join
right:
  call void @right_call()
  br label %join
join:
  %test = phi i1 [ %x, %left ], [ %y, %right ]
  br i1 %test, label %yes, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
yes:
  call void @a()
  ret void
no:
  call void @b()
  ret void
}

; A positive predecessor hint still describes the duplicate's block after an
; unconditional split and merge. It does not classify its new branch decision.
; CHECK-LABEL: define void @annotated_predecessor(
; CHECK: br i1 %x, label %yes, label %no, !prof ![[W]], !block.uniformity.profile ![[U:[0-9]+]]{{$}}
; CHECK: br i1 %y, label %yes, label %no, !prof ![[W]]{{$}}
define void @annotated_predecessor(i1 %dispatch, i1 %x, i1 %y) !uniformity.profile !0 {
entry:
  br i1 %dispatch, label %left, label %right
left:
  call void @left_call()
  br label %join, !block.uniformity.profile !0
right:
  call void @right_call()
  br label %join
join:
  %test = phi i1 [ %x, %left ], [ %y, %right ]
  br i1 %test, label %yes, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
yes:
  call void @a()
  ret void
no:
  call void @b()
  ret void
}

; An unchanged branch retains both classifications.
; CHECK-LABEL: define void @unchanged(
; CHECK: br i1 %cond, label %yes, label %no, !prof ![[W]], !block.uniformity.profile ![[U]], !branch.uniformity.profile ![[U]]{{$}}
define void @unchanged(i1 %cond) !uniformity.profile !0 {
entry:
  br i1 %cond, label %yes, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
yes:
  call void @a()
  ret void
no:
  call void @b()
  ret void
}

declare void @left_call()
declare void @right_call()
declare void @a()
declare void @b()

; CHECK: ![[W]] = !{!"branch_weights", i32 90, i32 10}
!0 = !{}
!1 = !{!"branch_weights", i32 90, i32 10}
