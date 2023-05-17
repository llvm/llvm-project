; RUN: opt < %s -passes=indvars -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

declare void @use(i64 %x)

; CHECK-LABEL: @foo
define void @foo() {
entry:
  br label %L1_header

L1_header:
  br label %L2_header

; CHECK: L2_header:
; CHECK: %[[INDVAR:.*]] = phi i64
; CHECK: %[[TRUNC:.*]] = trunc i64 %[[INDVAR]] to i32
L2_header:
  %i = phi i32 [ 0, %L1_header ], [ %i_next, %L2_latch ]
  %i_prom = sext i32 %i to i64
  call void @use(i64 %i_prom)
  br label %L3_header

L3_header:
  br i1 undef, label %L3_latch, label %L2_exiting_1

L3_latch:
  br i1 undef, label %L3_header, label %L2_exiting_2

L2_exiting_1:
  br i1 undef, label %L2_latch, label %L1_latch

L2_exiting_2:
  br i1 undef, label %L2_latch, label %L1_latch

L2_latch:
  %i_next = add nsw i32 %i, 1
  br label %L2_header

L1_latch:
; CHECK: L1_latch:
; CHECK: %i_lcssa = phi i32 [ %[[TRUNC]], %L2_exiting_1 ], [ %[[TRUNC]], %L2_exiting_2 ]

  %i_lcssa = phi i32 [ %i, %L2_exiting_1 ], [ %i, %L2_exiting_2 ]
  br i1 undef, label %exit, label %L1_header

exit:
  ret void
}
