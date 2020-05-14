; RUN: opt < %s -passes=partial-inliner -S 2>&1| FileCheck %s

define i32 @callee1(i32 %arg) {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb2

bb1:
  br i1 undef, label %bb4, label %bb2

bb2:
  br i1 undef, label %bb4, label %bb5

bb4:
  %xx1 = phi i32 [ 1, %bb1 ], [ 9, %bb2 ]
  %xx2 = phi i32 [ 1, %bb1 ], [ 9, %bb2 ]
  %xx3 = phi i32 [ 1, %bb1 ], [ 9, %bb2 ]
  tail call void (...) @extern() #2
  br label %bb5

bb5:
  %tmp6 = phi i32 [ 1, %bb2 ], [ 9, %bb4 ]
  ret i32 %tmp6
}

declare void @extern(...)

define i32 @caller1(i32 %arg) {
bb:
;; partial inliner inlines callee to caller.
; CHECK-LABEL: @caller1
; CHECK: br i1
; CHECK: br i1
; CHECK-NOT: call i32 @callee1(
  %tmp = tail call i32 @callee1(i32 %arg)
  ret i32 %tmp
}

define i32 @caller2(i32 %arg) #0 {
bb:
;; partial inliner won't inline callee to caller because they have
;; incompatible attributes.
; CHECK-LABEL: @caller2
; CHECK: call i32 @callee1(
  %tmp = tail call i32 @callee1(i32 %arg)
  ret i32 %tmp
}

attributes #0 = { "use-sample-profile" }
