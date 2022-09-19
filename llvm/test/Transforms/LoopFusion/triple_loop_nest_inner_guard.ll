; RUN: opt -S -loop-fusion < %s 2>&1 | FileCheck %s
; XFAIL: *
; REQUIRES: asserts
; Fails due to incorrect cached loop disposition.

; Verify that LoopFusion can fuse two triple-loop nests with guarded inner
; loops. Loops are in canonical form.

@a = common global [10 x [10 x [10 x i32]]] zeroinitializer
@b = common global [10 x [10 x [10 x i32]]] zeroinitializer
@c = common global [10 x [10 x [10 x i32]]] zeroinitializer

; CHECK-LABEL: @triple_loop_nest_inner_guard
; CHECK: br i1 %{{.*}}, label %[[OUTER_PH:outer1.ph]], label %[[FUNC_EXIT:func_exit]]

; CHECK: [[OUTER_PH]]:
; CHECK: br label %[[OUTER_BODY_MIDDLE_GUARD:outer1.body.middle1.guard]]

; CHECK: [[OUTER_BODY_MIDDLE_GUARD]]:
; CHECK: br i1 %{{.*}}, label %[[MIDDLE_PH:middle1.ph]], label %[[OUTER_LATCH:outer2.latch]]

; CHECK: [[MIDDLE_PH]]:
; CHECK-NEXT: br label %[[MIDDLE_BODY_INNER_GUARD:middle1.body.inner1.guard]]

; CHECK: [[MIDDLE_BODY_INNER_GUARD]]:
; CHECK: br i1 %{{.*}}, label %[[INNER_PH:inner1.ph]], label %[[MIDDLE_LATCH:middle2.latch]]

; CHECK: [[INNER_PH]]:
; CHECK-NEXT: br label %[[INNER_BODY:inner1.body]]

; CHECK: [[INNER_BODY]]:
; First loop body.
; CHECK: load
; CHECK: add
; CHECK: store
; Second loop body.
; CHECK: load
; CHECK: mul
; CHECK: store
; CHECK: br i1 %{{.*}}, label %[[INNER_EXIT:inner2.exit]], label %[[INNER_BODY:inner1.body]]

; CHECK: [[INNER_EXIT]]:
; CHECK-NEXT: br label %[[MIDDLE_LATCH:middle2.latch]]

; CHECK: [[MIDDLE_LATCH]]:
; CHECK: br i1 %{{.*}}, label %[[MIDDLE_EXIT:middle2.exit]], label %[[MIDDLE_BODY_INNER_GUARD]]

; CHECK: [[MIDDLE_EXIT]]:
; CHECK-NEXT: br label %[[OUTER_LATCH:outer2.latch]]

; CHECK: [[OUTER_LATCH]]:
; CHECK: br i1 %{{.*}}, label %[[OUTER_EXIT:outer2.exit]], label %[[OUTER_BODY_MIDDLE_GUARD]]

; CHECK: [[OUTER_EXIT]]:
; CHECK-NEXT: br label %[[FUNC_EXIT:func_exit]]

; CHECK: [[FUNC_EXIT]]:
; CHECK-NEXT: ret

define i32 @triple_loop_nest_inner_guard(i32 %m, i32 %n, i32 %M, i32 %N) {
entry:
  %cmp101 = icmp sgt i32 %m, 0
  br i1 %cmp101, label %outer1.ph, label %func_exit

outer1.ph:
  %cmp298 = icmp sgt i32 %n, 0
  %cmp696 = icmp sgt i32 %M, 0
  %wide.trip.count122 = zext i32 %m to i64
  %wide.trip.count118 = zext i32 %n to i64
  %wide.trip.count114 = zext i32 %M to i64
  br label %outer1.body.middle1.guard

outer1.body.middle1.guard:
  %iv120 = phi i64 [ 0, %outer1.ph ], [ %iv.next121, %outer1.latch ]
  br i1 %cmp298, label %middle1.ph, label %outer1.latch

middle1.ph:
  br label %middle1.body.inner1.guard

middle1.body.inner1.guard:
  %iv116 = phi i64 [ %iv.next117, %middle1.latch ], [ 0, %middle1.ph ]
  br i1 %cmp696, label %inner1.ph, label %middle1.latch

inner1.ph:
  br label %inner1.body

inner1.body:
  %iv112 = phi i64 [ %iv.next113, %inner1.body ], [ 0, %inner1.ph ]
  %idx12 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* @a, i64 0, i64 %iv120, i64 %iv116, i64 %iv112
  %0 = load i32, i32* %idx12
  %add = add nsw i32 %0, 2
  %idx18 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* @b, i64 0, i64 %iv120, i64 %iv116, i64 %iv112
  store i32 %add, i32* %idx18
  %iv.next113 = add nuw nsw i64 %iv112, 1
  %exitcond115 = icmp eq i64 %iv.next113, %wide.trip.count114
  br i1 %exitcond115, label %inner1.exit, label %inner1.body

inner1.exit:
  br label %middle1.latch

middle1.latch:
  %iv.next117 = add nuw nsw i64 %iv116, 1
  %exitcond119 = icmp eq i64 %iv.next117, %wide.trip.count118
  br i1 %exitcond119, label %middle1.exit, label %middle1.body.inner1.guard

middle1.exit:
  br label %outer1.latch

outer1.latch:
  %iv.next121 = add nuw nsw i64 %iv120, 1
  %exitcond123 = icmp eq i64 %iv.next121, %wide.trip.count122
  br i1 %exitcond123, label %outer2.ph, label %outer1.body.middle1.guard

outer2.ph:
  br label %outer2.middle2.guard

outer2.middle2.guard:
  %iv108 = phi i64 [ %iv.next109, %outer2.latch ], [ 0, %outer2.ph ]
  br i1 %cmp298, label %middle2.ph, label %outer2.latch

middle2.ph:
  br label %middle2.body.inner2.guard

middle2.body.inner2.guard:
  %iv104 = phi i64 [ %iv.next105, %middle2.latch ], [ 0, %middle2.ph ]
  br i1 %cmp696, label %inner2.ph, label %middle2.latch

inner2.ph:
  br label %inner2.body

inner2.body:
  %iv = phi i64 [ %iv.next, %inner2.body ], [ 0, %inner2.ph ]
  %idx45 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* @a, i64 0, i64 %iv108, i64 %iv104, i64 %iv
  %1 = load i32, i32* %idx45
  %mul = shl nsw i32 %1, 1
  %idx51 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* @c, i64 0, i64 %iv108, i64 %iv104, i64 %iv
  store i32 %mul, i32* %idx51
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %wide.trip.count114
  br i1 %exitcond, label %inner2.exit, label %inner2.body

inner2.exit:
  br label %middle2.latch

middle2.latch:
  %iv.next105 = add nuw nsw i64 %iv104, 1
  %exitcond107 = icmp eq i64 %iv.next105, %wide.trip.count118
  br i1 %exitcond107, label %middle2.exit, label %middle2.body.inner2.guard

middle2.exit:
  br label %outer2.latch

outer2.latch:
  %iv.next109 = add nuw nsw i64 %iv108, 1
  %exitcond111 = icmp eq i64 %iv.next109, %wide.trip.count122
  br i1 %exitcond111, label %outer2.exit, label %outer2.middle2.guard

outer2.exit:
  br label %func_exit

func_exit:
  ret i32 undef
}
