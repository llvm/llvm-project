; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define i32 @slt_trip_count_with_range(ptr %ptr0, ptr %ptr1) {
; CHECK-LABEL: slt_trip_count_with_range
 entry:
  %limit = load i32, ptr %ptr0, !range !0
  br label %loop

 loop:
; CHECK: Loop %loop: constant max backedge-taken count is 98
  %index = phi i32 [ 0, %entry ], [ %index.inc, %loop ]
  %index.inc = add i32 %index, 1
  %continue = icmp slt i32 %index.inc, %limit
  br i1 %continue, label %loop, label %loop.exit

 loop.exit:
  ret i32 0
}

define i32 @ult_trip_count_with_range(ptr %ptr0, ptr %ptr1) {
; CHECK-LABEL: ult_trip_count_with_range
 entry:
  %limit = load i32, ptr %ptr0, !range !0
  br label %loop

 loop:
; CHECK: Loop %loop: constant max backedge-taken count is 98
  %index = phi i32 [ 0, %entry ], [ %index.inc, %loop ]
  %index.inc = add i32 %index, 1
  %continue = icmp ult i32 %index.inc, %limit
  br i1 %continue, label %loop, label %loop.exit

 loop.exit:
  ret i32 0
}

!0 = !{i32 1, i32 100}
