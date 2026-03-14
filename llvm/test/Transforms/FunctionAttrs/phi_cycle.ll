; RUN: opt -passes=function-attrs -S < %s

; Regression test for a null-returning bug of getUnderlyingObjectAggressive().
; This should not crash.
define void @phi_cycle() {
bb:
  unreachable

bb1:                                              ; preds = %bb17
  br label %bb2

bb2:                                              ; preds = %bb5, %bb1
  %phi = phi ptr [ %phi6, %bb1 ], [ %phi6, %bb5 ]
  br i1 poison, label %bb4, label %bb3

bb3:                                              ; preds = %bb2
  %getelementptr = getelementptr inbounds i8, ptr %phi, i32 poison
  br label %bb5

bb4:                                              ; preds = %bb2
  br label %bb7

bb5:                                              ; preds = %bb15, %bb3
  %phi6 = phi ptr [ %getelementptr, %bb3 ], [ %phi16, %bb15 ]
  br i1 poison, label %bb17, label %bb2

bb7:                                              ; preds = %bb15, %bb4
  %phi8 = phi ptr [ %phi, %bb4 ], [ %phi16, %bb15 ]
  br i1 poison, label %bb11, label %bb9

bb9:                                              ; preds = %bb7
  %getelementptr10 = getelementptr inbounds i8, ptr %phi8, i32 1
  store i8 poison, ptr %phi8, align 1
  br label %bb15

bb11:                                             ; preds = %bb7
  br i1 poison, label %bb13, label %bb12

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb12, %bb11
  %getelementptr14 = getelementptr inbounds i8, ptr %phi8, i32 poison
  br label %bb15

bb15:                                             ; preds = %bb13, %bb9
  %phi16 = phi ptr [ %getelementptr14, %bb13 ], [ %getelementptr10, %bb9 ]
  br i1 poison, label %bb5, label %bb7

bb17:                                             ; preds = %bb5
  br label %bb1
}
