; REQUIRES: asserts
; RUN: opt -jump-threading -aa-pipeline basic-aa -S -disable-output %s
; RUN: opt -passes=jump-threading -aa-pipeline basic-aa -S -disable-output %s

define void @foo(ptr %arg1, ptr %arg2) {
bb:
  br label %bb1

bb1:
  %tmp = phi i1 [ 0, %bb24 ], [ 1, %bb ]
  br i1 %tmp, label %bb9, label %bb24


bb9:
  br i1 %tmp, label %bb8, label %bb20

bb8:
  ret void

bb13:
  br label %bb14

bb14:
  %tmp15 = phi ptr [ %tmp21, %bb20 ], [ %arg2, %bb13 ]
  %tmp16 = phi ptr [ %tmp22, %bb20 ], [ %arg1, %bb13 ]
  store atomic i32 0, ptr %tmp15 unordered, align 4
  %tmp17 = load atomic ptr, ptr %tmp16 unordered, align 8
  %tmp18 = icmp eq ptr %tmp17, null
  br i1 %tmp18, label %bb25, label %bb19

bb19:
  ret void

bb20:
  %tmp21 = phi ptr [ %arg2, %bb9 ]
  %tmp22 = phi ptr [ %arg1, %bb9 ]
  br label %bb14

bb24:
  br label %bb1

bb25:
  ret void
}
