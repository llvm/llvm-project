; RUN: opt -passes=jump-threading -S < %s | FileCheck %s
; CHECK: @main

%struct.wobble = type { i8 }

define i32 @main() local_unnamed_addr personality ptr undef {
bb12:
  br i1 false, label %bb13, label %bb28

bb13:                                             ; preds = %bb12
  br label %bb14

bb14:                                             ; preds = %bb26, %bb13
  %tmp15 = phi ptr [ %tmp27, %bb26 ], [ undef, %bb13 ]
  %tmp16 = icmp slt i32 5, undef
  %tmp17 = select i1 false, i1 true, i1 %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14
  br i1 %tmp17, label %bb19, label %bb21

bb19:                                             ; preds = %bb18
  %tmp20 = or i32 undef, 4
  br label %bb21

bb21:                                             ; preds = %bb19, %bb18
  %tmp22 = load i8, ptr %tmp15, align 1
  br label %bb23

bb23:                                             ; preds = %bb21
  br i1 %tmp17, label %bb24, label %bb25

bb24:                                             ; preds = %bb23
  br label %bb25

bb25:                                             ; preds = %bb24, %bb23
  invoke void undef(ptr undef, i32 0, i32 undef, i8 %tmp22)
          to label %bb26 unwind label %bb33

bb26:                                             ; preds = %bb25
  %tmp27 = getelementptr inbounds i8, ptr %tmp15, i64 1
  br label %bb14

bb28:                                             ; preds = %bb12
  unreachable

bb33:                                             ; preds = %bb25
  %tmp34 = landingpad { ptr, i32 }
          cleanup
  unreachable
}
