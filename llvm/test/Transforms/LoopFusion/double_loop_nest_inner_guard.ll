; RUN: opt -S -passes=loop-fusion < %s 2>&1 | FileCheck %s

; Verify that LoopFusion can fuse two double-loop nests with guarded inner
; loops. Loops are in canonical form.

@a = common global [10 x [10 x i32]] zeroinitializer
@b = common global [10 x [10 x i32]] zeroinitializer
@c = common global [10 x [10 x i32]] zeroinitializer

; CHECK-LABEL: @double_loop_nest_inner_guard
; CHECK: br i1 %{{.*}}, label %[[OUTER_PH:outer1.ph]], label %[[FUNC_EXIT:func_exit]]

; CHECK: [[OUTER_PH]]:
; CHECK: br label %[[OUTER_BODY_INNER_GUARD:outer1.body.inner.guard]]

; CHECK: [[OUTER_BODY_INNER_GUARD]]:
; CHECK: br i1 %{{.*}}, label %[[INNER_PH:inner1.ph]], label %[[OUTER_LATCH:outer2.latch]]

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
; CHECK-NEXT: br label %[[OUTER_LATCH:outer2.latch]]

; CHECK: [[OUTER_LATCH]]:
; CHECK: br i1 %{{.*}}, label %[[OUTER_EXIT:outer2.exit]], label %[[OUTER_BODY_INNER_GUARD]]

; CHECK: [[OUTER_EXIT]]:
; CHECK-NEXT: br label %[[FUNC_EXIT:func_exit]]

; CHECK: [[FUNC_EXIT]]:
; CHECK-NEXT: ret

define i32 @double_loop_nest_inner_guard(i32 %m, i32 %n, i32 %M, i32 %N) {
entry:
  %cmp63 = icmp sgt i32 %m, 0
  br i1 %cmp63, label %outer1.ph, label %func_exit

outer1.ph:
  %cmp261 = icmp sgt i32 %n, 0
  %wide.trip.count76 = zext i32 %m to i64
  %wide.trip.count72 = zext i32 %n to i64
  br label %outer1.body.inner.guard

outer1.body.inner.guard:
  %iv74 = phi i64 [ 0, %outer1.ph ], [ %iv.next75, %outer1.latch ]
  br i1 %cmp261, label %inner1.ph, label %outer1.latch

inner1.ph:
  br label %inner1.body

inner1.body:
  %iv70 = phi i64 [ %iv.next71, %inner1.body ], [ 0, %inner1.ph ]
  %idx6 = getelementptr inbounds [10 x [10 x i32]], ptr @a, i64 0, i64 %iv74, i64 %iv70
  %0 = load i32, ptr %idx6
  %add = add nsw i32 %0, 2
  %idx10 = getelementptr inbounds [10 x [10 x i32]], ptr @b, i64 0, i64 %iv74, i64 %iv70
  store i32 %add, ptr %idx10
  %iv.next71 = add nuw nsw i64 %iv70, 1
  %exitcond73 = icmp eq i64 %iv.next71, %wide.trip.count72
  br i1 %exitcond73, label %inner1.exit, label %inner1.body

inner1.exit:
  br label %outer1.latch

outer1.latch:
  %iv.next75 = add nuw nsw i64 %iv74, 1
  %exitcond77 = icmp eq i64 %iv.next75, %wide.trip.count76
  br i1 %exitcond77, label %outer2.ph, label %outer1.body.inner.guard

outer2.ph:
  br label %outer2.body.inner.guard

outer2.body.inner.guard:
  %iv66 = phi i64 [ %iv.next67, %outer2.latch ], [ 0, %outer2.ph ]
  br i1 %cmp261, label %inner2.ph, label %outer2.latch

inner2.ph:
  br label %inner2.body

inner2.body:
  %iv = phi i64 [ %iv.next, %inner2.body ], [ 0, %inner2.ph ]
  %idx27 = getelementptr inbounds [10 x [10 x i32]], ptr @a, i64 0, i64 %iv66, i64 %iv
  %1 = load i32, ptr %idx27
  %mul = shl nsw i32 %1, 1
  %idx31 = getelementptr inbounds [10 x [10 x i32]], ptr @c, i64 0, i64 %iv66, i64 %iv
  store i32 %mul, ptr %idx31
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %wide.trip.count72
  br i1 %exitcond, label %inner2.exit, label %inner2.body

inner2.exit:
  br label %outer2.latch

outer2.latch:
  %iv.next67 = add nuw nsw i64 %iv66, 1
  %exitcond69 = icmp eq i64 %iv.next67, %wide.trip.count76
  br i1 %exitcond69, label %outer2.exit, label %outer2.body.inner.guard

outer2.exit:
  br label %func_exit

func_exit:
  ret i32 undef
}
