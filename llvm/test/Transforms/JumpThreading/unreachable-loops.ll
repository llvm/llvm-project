; RUN: opt -jump-threading -S < %s | FileCheck %s
; RUN: opt -passes=jump-threading -S < %s | FileCheck %s
; Check the unreachable loop won't cause infinite loop
; in jump-threading when it tries to update the predecessors'
; profile metadata from a phi node.

define void @unreachable_single_bb_loop() {
; CHECK-LABEL: @unreachable_single_bb_loop()
bb:
  %tmp = call i32 @a()
  %tmp1 = icmp eq i32 %tmp, 1
  br i1 %tmp1, label %bb5, label %bb8

; unreachable single bb loop.
bb2:                                              ; preds = %bb2
  %tmp4 = icmp ne i32 %tmp, 1
  switch i1 %tmp4, label %bb2 [
    i1 0, label %bb5
    i1 1, label %bb8
  ]

bb5:                                              ; preds = %bb2, %bb
  %tmp6 = phi i1 [ %tmp1, %bb ], [ false, %bb2 ]
  br i1 %tmp6, label %bb8, label %bb7, !prof !0

bb7:                                              ; preds = %bb5
  br label %bb8

bb8:                                              ; preds = %bb8, %bb7, %bb5, %bb2
  ret void
}

define void @unreachable_multi_bbs_loop() {
; CHECK-LABEL: @unreachable_multi_bbs_loop()
bb:
  %tmp = call i32 @a()
  %tmp1 = icmp eq i32 %tmp, 1
  br i1 %tmp1, label %bb5, label %bb8

; unreachable two bbs loop.
bb3:                                              ; preds = %bb2
  br label %bb2

bb2:                                              ; preds = %bb3
  %tmp4 = icmp ne i32 %tmp, 1
  switch i1 %tmp4, label %bb3 [
    i1 0, label %bb5
    i1 1, label %bb8
  ]

bb5:                                              ; preds = %bb2, %bb
  %tmp6 = phi i1 [ %tmp1, %bb ], [ false, %bb2 ]
  br i1 %tmp6, label %bb8, label %bb7, !prof !0

bb7:                                              ; preds = %bb5
  br label %bb8

bb8:                                              ; preds = %bb8, %bb7, %bb5, %bb2
  ret void
}
declare i32 @a()

!0 = !{!"branch_weights", i32 2146410443, i32 1073205}
