; RUN: opt -S -passes=simplifycfg -verify-each < %s | FileCheck %s

; Cover both condition folding implementations. The bypassed block survives on
; the right path, where it must lose its old block and branch classifications.

declare void @a()
declare void @b()
declare void @left_call()
declare void @right_call()

; CHECK-LABEL: define void @computed(
; CHECK: br i1 %or.cond, label %yes, label %no, !prof ![[COMPUTED:[0-9]+]], !block.uniformity.profile ![[U:[0-9]+]]{{$}}
; CHECK: br i1 %y.old, label %yes, label %no, !prof ![[ORIGINAL:[0-9]+]]{{$}}
define void @computed(i1 %dispatch, i1 %x, i32 %value) !uniformity.profile !0 {
entry:
  br i1 %dispatch, label %left, label %right
left:
  call void @left_call()
  br i1 %x, label %inner, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
right:
  call void @right_call()
  br label %inner
inner:
  %y = icmp eq i32 %value, 0
  br i1 %y, label %yes, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
yes:
  call void @a()
  ret void
no:
  call void @b()
  ret void
}

; CHECK-LABEL: define void @empty(
; CHECK: br i1 %brmerge, label %no, label %yes, !prof ![[EMPTY:[0-9]+]], !block.uniformity.profile ![[U]]{{$}}
; CHECK: br i1 %y, label %yes, label %no, !prof ![[ORIGINAL]]{{$}}
define void @empty(i1 %dispatch, i1 %x, i1 %y) !uniformity.profile !0 {
entry:
  br i1 %dispatch, label %left, label %right
left:
  call void @left_call()
  br i1 %x, label %inner, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
right:
  call void @right_call()
  br label %inner
inner:
  br i1 %y, label %yes, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
yes:
  call void @a()
  ret void
no:
  call void @b()
  ret void
}
; CHECK-DAG: ![[COMPUTED]] = !{!"branch_weights", i32 8100, i32 1900}
; CHECK-DAG: ![[ORIGINAL]] = !{!"branch_weights", i32 90, i32 10}
; CHECK-DAG: ![[EMPTY]] = !{!"branch_weights", i32 1900, i32 8100}
!0 = !{}
!1 = !{!"branch_weights", i32 90, i32 10}
