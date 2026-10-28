; RUN: opt -S -passes=simplifycfg -verify-each < %s | FileCheck %s

; Folding two predicates must drop the old branch hint, but retain the
; unchanged entry block hint and recompute branch weights.
; CHECK-LABEL: define void @fold(
; CHECK: br i1 %brmerge, label %no, label %yes, !prof ![[FOLD:[0-9]+]], !block.uniformity.profile ![[U:[0-9]+]]{{$}}
define void @fold(i1 %x, i1 %y) !uniformity.profile !0 {
entry:
  br i1 %x, label %inner, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
inner:
  br i1 %y, label %yes, label %no, !prof !1
yes:
  call void @a()
  ret void
no:
  call void @b()
  ret void
}

; Missing outer weights do not justify preserving branch uniformity.
; CHECK-LABEL: define void @fold_proxy(
; CHECK: br i1 %brmerge, label %no, label %yes, !prof !{{[0-9]+}}, !block.uniformity.profile ![[U]]{{$}}
define void @fold_proxy(i1 %x, i1 %y) !uniformity.profile !0 {
entry:
  br i1 %x, label %inner, label %no, !block.uniformity.profile !0, !branch.uniformity.profile !0
inner:
  br i1 %y, label %yes, label %no, !prof !2, !block.uniformity.profile !0
yes:
  call void @a()
  ret void
no:
  call void @b()
  ret void
}

; Preserve block execution information through branch replacement and merge.
; CHECK-LABEL: define void @constant(
; CHECK: call void @a()
; CHECK-NEXT: ret void, !block.uniformity.profile ![[U]]{{$}}
define void @constant(i1 %dispatch) !uniformity.profile !0 {
entry:
  call void @a()
  br i1 true, label %yes, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
yes:
  ret void
no:
  call void @a()
  ret void
}

; Switch weights are default/case; branch weights are true/false.
; A new conditional branch must not acquire a branch-uniformity hint.
; CHECK-LABEL: define void @switch_one(
; CHECK: br i1 %cond, label %yes, label %no, !prof ![[SWITCH:[0-9]+]], !block.uniformity.profile ![[U]]{{$}}
define void @switch_one(i32 %x) !uniformity.profile !0 {
entry:
  switch i32 %x, label %no [
    i32 0, label %yes
  ], !prof !3, !block.uniformity.profile !0
yes:
  call void @a()
  ret void
no:
  call void @b()
  ret void
}

; Folding a constant switch retains block metadata through the return merge.
; CHECK-LABEL: define void @constant_switch(
; CHECK: call void @a()
; CHECK-NEXT: ret void, !block.uniformity.profile ![[U]]{{$}}
define void @constant_switch() !uniformity.profile !0 {
entry:
  call void @a()
  switch i32 0, label %no [i32 0, label %yes], !prof !3, !block.uniformity.profile !0
yes:
  ret void
no:
  call void @b()
  ret void
}

; An unknown block must not become annotated merely because profile data exists.
; CHECK-LABEL: define void @unknown(
; CHECK: call void @a()
; CHECK-NEXT: ret void{{$}}
define void @unknown() !uniformity.profile !0 {
entry:
  call void @a()
  br i1 true, label %yes, label %no
yes:
  ret void
no:
  call void @b()
  ret void
}

; Both arms lead to the same block: the branch hint becomes obsolete, while
; the predecessor block's classification survives replacement and merging.
; CHECK-LABEL: define void @same_destination(
; CHECK: call void @a()
; CHECK-NEXT: ret void, !block.uniformity.profile ![[U]]{{$}}
define void @same_destination(i1 %cond) !uniformity.profile !0 {
entry:
  call void @a()
  br i1 %cond, label %exit, label %exit, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
exit:
  ret void
}

declare void @a()
declare void @b()

; CHECK-DAG: ![[U]] = !{}
; CHECK-DAG: ![[FOLD]] = !{!"branch_weights", i32 1900, i32 8100}
; CHECK-DAG: ![[SWITCH]] = !{!"branch_weights", i32 90, i32 10}
!0 = !{}
!1 = !{!"branch_weights", i32 90, i32 10}
!2 = !{!"branch_weights", i32 50, i32 50}
!3 = !{!"branch_weights", i32 10, i32 90}
