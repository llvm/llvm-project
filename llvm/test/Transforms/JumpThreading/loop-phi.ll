; RUN: opt < %s -jump-threading -S -jump-threading-across-loop-headers | FileCheck %s

; Make sure we correctly distinguish between %tmp15 and %tmp16 when we clone
; body2.

; CHECK:      body2.thread:
; CHECK-NEXT: %tmp163 = add i32 %tmp165, 1
; CHECK-NEXT: br label %latch1

; CHECK:      latch1:
; CHECK-NEXT: %tmp165 = phi i32 [ %tmp163, %body2.thread ], [ %tmp16, %body2 ]
; CHECK-NEXT: %tmp154 = phi i32 [ %tmp165, %body2.thread ], [ %tmp15, %body2 ]

define i32 @test(i1 %ARG1, i1 %ARG2, i32 %n) {
entry:
  br label %head1

head1:                                            ; preds = %entry, %body1
  %tmp = phi i32 [ 0, %entry ], [ %tmp16, %body1 ]
  %tmp3 = phi i32 [ 0, %entry ], [ %tmp16, %body1 ]
  %tmp4 = phi i32 [ 0, %entry ], [ %tmp16, %body1 ]
  br i1 %ARG1, label %exit, label %body2

body1:                                            ; preds = %latch1
  %tmp12 = icmp sgt i32 %tmp16, 1
  br i1 %tmp12, label %body2, label %head1

body2:                                            ; preds = %head1, %body1
  %tmp14 = phi i32 [ %tmp16, %body1 ], [ %tmp, %head1 ]
  %tmp15 = phi i32 [ %tmp16, %body1 ], [ %tmp3, %head1 ]
  %tmp16 = add i32 %tmp14, 1
  br i1 %ARG2, label %exit, label %latch1

latch1:                                           ; preds = %body2
  %tmp18 = icmp sgt i32 %tmp16, %n
  br i1 %tmp18, label %exit, label %body1

exit:                                             ; preds = %latch1, %body2, %head1
  %rc = phi i32 [ %tmp15, %body2 ], [ %tmp15, %latch1 ], [ -1, %head1 ]
  ret i32 %rc
}
