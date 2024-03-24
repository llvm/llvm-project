; RUN: opt -S -passes=lower-switch %s | FileCheck %s

; CHECK-LABEL: @phi_in_dead_block(
; CHECK-NOT: switch
define void @phi_in_dead_block() {
bb:
  br i1 undef, label %bb2, label %bb3

bb1:                                              ; No predecessors!
  switch i32 undef, label %bb2 [
    i32 9, label %bb3
  ]

bb2:                                              ; preds = %bb1, %bb
  %tmp = phi i64 [ undef, %bb1 ], [ undef, %bb ]
  unreachable

bb3:                                              ; preds = %bb1, %bb
  unreachable
}

; CHECK-LABEL: @phi_in_dead_block_br_to_self(
; CHECK-NOT: switch
define void @phi_in_dead_block_br_to_self() {
bb:
  br i1 undef, label %bb2, label %bb3

bb1:                                              ; No predecessors!
  switch i32 undef, label %bb2 [
    i32 9, label %bb3
    i32 10, label %bb1
  ]

bb2:                                              ; preds = %bb1, %bb
  %tmp = phi i64 [ undef, %bb1 ], [ undef, %bb ]
  unreachable

bb3:                                              ; preds = %bb1, %bb
  unreachable
}
