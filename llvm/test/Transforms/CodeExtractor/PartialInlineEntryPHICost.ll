; RUN: opt < %s -passes=partial-inliner -S | FileCheck %s

; Check that we do not overcompute the outlined region cost, where the PHIs in
; the outlined region entry (BB4) are moved outside the region by CodeExtractor.

define i32 @bar(i32 %arg) {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb2

bb1:
  br i1 undef, label %bb4, label %bb2

bb2:                                              ; preds = %bb, %bb1
  br i1 undef, label %bb4, label %bb5

bb4:                                              ; preds = %bb1, %bb2
  %xx1 = phi i32 [ 1, %bb1 ], [ 9, %bb2 ]
  %xx2 = phi i32 [ 1, %bb1 ], [ 9, %bb2 ]
  %xx3 = phi i32 [ 1, %bb1 ], [ 9, %bb2 ]
  tail call void (...) @foo() #2
  br label %bb5

bb5:                                              ; preds = %bb4, %bb2
  %tmp6 = phi i32 [ 1, %bb2 ], [ 9, %bb4 ]
  ret i32 %tmp6
}

declare void @foo(...)

define i32 @dummy_caller(i32 %arg) {
bb:
; CHECK-LABEL: @dummy_caller
; CHECK: br i1
; CHECK: br i1
; CHECK: call void @bar.1.
  %tmp = tail call i32 @bar(i32 %arg)
  ret i32 %tmp
}
