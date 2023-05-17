; RUN: opt -S < %s -passes=jump-threading -jump-threading-across-loop-headers | FileCheck %s

; CHECK-LABEL: @foo
; Just check that we don't hang on this test.

define void @foo(i32 %a) {
bb_entry:
  br label %bb_header

bb_header:
  %b = phi i32 [ %c, %bb_header ], [ 0, %bb_body1 ], [ 2, %bb_body2 ], [ 0, %bb_entry ]
  %c = add nuw nsw i32 %b, 1
  %d = icmp ult i32 %c, 6
  br i1 %d, label %bb_header, label %bb_body1

bb_body1:
  %e = icmp eq i32 %a, 0
  br i1 %e, label %bb_body2, label %bb_header

bb_body2:
  br label %bb_header
}
