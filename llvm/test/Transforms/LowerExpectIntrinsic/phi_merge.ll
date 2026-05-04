; RUN: opt -passes=lower-expect  -S -o - < %s | FileCheck %s
; RUN: opt -S -passes='function(lower-expect)' < %s | FileCheck %s

; The C case
; if (__builtin_expect((x > goo() && y > hoo() && z > too()), 1))
; For the above case, all 3 branches should be annotated.
;
; if (__builtin_expect((x > goo() && y > hoo() && z > too()), 0))
; For the above case, we don't have enough information, so
; only the last branch is annotated.

define void @foo(i32 %arg, i32 %arg1, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: void @foo
bb:
  %tmp8 = call i32  @goo()
  %tmp9 = icmp sgt i32 %tmp8, %arg
  br i1 %tmp9, label %bb10, label %bb18
; CHECK: !prof [[WEIGHT:![0-9]+]]

bb10:                                             ; preds = %bb
  %tmp12 = call i32  @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br i1 %tmp13, label %bb14, label %bb18
; CHECK: br i1 %tmp13, {{.*}}!prof [[WEIGHT]]

bb14:                                             ; preds = %bb10
  %tmp16 = call i32  @too()
  %tmp17 = icmp sgt i32 %arg2, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14, %bb10, %bb
  %tmp19 = phi i1 [ false, %bb10 ], [ false, %bb ], [ %tmp17, %bb14 ]
  %tmp20 = xor i1 %tmp19, true
  %tmp21 = xor i1 %tmp20, true
  %tmp22 = zext i1 %tmp21 to i32
  %tmp23 = sext i32 %tmp22 to i64
  %tmp24 = call i64 @llvm.expect.i64(i64 %tmp23, i64 1)
  %tmp25 = icmp ne i64 %tmp24, 0
  br i1 %tmp25, label %bb26, label %bb28
; CHECK: br i1 %tmp25,{{.*}}!prof [[WEIGHT]]

bb26:                                             ; preds = %bb18
  %tmp27 = call i32  @goo()
  br label %bb30

bb28:                                             ; preds = %bb18
  %tmp29 = call i32  @hoo()
  br label %bb30

bb30:                                             ; preds = %bb28, %bb26
  ret void
}

define void @foo2(i32 %arg, i32 %arg1, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: void @foo2
bb:
  %tmp8 = call i32  @goo()
  %tmp9 = icmp sgt i32 %tmp8, %arg
  br i1 %tmp9, label %bb10, label %bb18
; CHECK:  br i1 %tmp9
; CHECK-NOT: !prof

bb10:                                             ; preds = %bb
  %tmp12 = call i32  @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br i1 %tmp13, label %bb14, label %bb18
; CHECK: br i1 %tmp13
; CHECK-NOT: !prof

bb14:                                             ; preds = %bb10
  %tmp16 = call i32 @too()
  %tmp17 = icmp sgt i32 %arg2, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14, %bb10, %bb
  %tmp19 = phi i1 [ false, %bb10 ], [ false, %bb ], [ %tmp17, %bb14 ]
  %tmp20 = xor i1 %tmp19, true
  %tmp21 = xor i1 %tmp20, true
  %tmp22 = zext i1 %tmp21 to i32
  %tmp23 = sext i32 %tmp22 to i64
  %tmp24 = call i64 @llvm.expect.i64(i64 %tmp23, i64 0)
  %tmp25 = icmp ne i64 %tmp24, 0
  br i1 %tmp25, label %bb26, label %bb28
; CHECK: br i1 %tmp25,{{.*}}!prof [[WEIGHT2:![0-9]+]]

bb26:                                             ; preds = %bb18
  %tmp27 = call i32 @goo()
  br label %bb30

bb28:                                             ; preds = %bb18
  %tmp29 = call i32 @hoo()
  br label %bb30

bb30:                                             ; preds = %bb28, %bb26
  ret void
}

define void @foo_i32(i32 %arg, i32 %arg1, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: void @foo_i32
bb:
  %tmp8 = call i32  @goo()
  %tmp9 = icmp sgt i32 %tmp8, %arg
  br i1 %tmp9, label %bb10, label %bb18
; CHECK: !prof [[WEIGHT]]

bb10:                                             ; preds = %bb
  %tmp12 = call i32 @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br i1 %tmp13, label %bb14, label %bb18
; CHECK: br i1 %tmp13, {{.*}}!prof [[WEIGHT]]

bb14:                                             ; preds = %bb10
  %tmp16 = call i32 @too()
  %tmp17 = icmp sgt i32 %arg2, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14, %bb10, %bb
  %tmp19 = phi i32 [ 5, %bb10 ], [ 5, %bb ], [ %tmp16, %bb14 ]
  %tmp23 = sext i32 %tmp19 to i64
  %tmp24 = call i64 @llvm.expect.i64(i64 %tmp23, i64 4)
  %tmp25 = icmp ne i64 %tmp24, 0
  br i1 %tmp25, label %bb26, label %bb28
; CHECK: br i1 %tmp25,{{.*}}!prof [[WEIGHT]]

bb26:                                             ; preds = %bb18
  %tmp27 = call i32 @goo()
  br label %bb30

bb28:                                             ; preds = %bb18
  %tmp29 = call i32 @hoo()
  br label %bb30

bb30:                                             ; preds = %bb28, %bb26
  ret void
}


define void @foo_i32_not_unlikely(i32 %arg, i32 %arg1, i32 %arg2, i32 %arg3)  {
; CHECK-LABEL: void @foo_i32_not_unlikely
bb:
  %tmp8 = call i32 @goo()
  %tmp9 = icmp sgt i32 %tmp8, %arg
  br i1 %tmp9, label %bb10, label %bb18
; CHECK: br i1 %tmp9
; CHECK-NOT: !prof

bb10:                                             ; preds = %bb
  %tmp12 = call i32 @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br i1 %tmp13, label %bb14, label %bb18
; CHECK: br i1 %tmp13
; CHECK-NOT: !prof

bb14:                                             ; preds = %bb10
  %tmp16 = call i32  @too()
  %tmp17 = icmp sgt i32 %arg2, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14, %bb10, %bb
  %tmp19 = phi i32 [ 4, %bb10 ], [ 4, %bb ], [ %tmp16, %bb14 ]
  %tmp23 = sext i32 %tmp19 to i64
  %tmp24 = call i64 @llvm.expect.i64(i64 %tmp23, i64 4)
  %tmp25 = icmp ne i64 %tmp24, 0
  br i1 %tmp25, label %bb26, label %bb28
; CHECK: br i1 %tmp25,{{.*}}!prof [[WEIGHT]]

bb26:                                             ; preds = %bb18
  %tmp27 = call i32  @goo()
  br label %bb30

bb28:                                             ; preds = %bb18
  %tmp29 = call i32 @hoo()
  br label %bb30

bb30:                                             ; preds = %bb28, %bb26
  ret void
}

define void @foo_i32_xor(i32 %arg, i32 %arg1, i32 %arg2, i32 %arg3)  {
; CHECK-LABEL: void @foo_i32_xor
bb:
  %tmp8 = call i32  @goo()
  %tmp9 = icmp sgt i32 %tmp8, %arg
  br i1 %tmp9, label %bb10, label %bb18
; CHECK: br i1 %tmp9,{{.*}}!prof [[WEIGHT]]

bb10:                                             ; preds = %bb
  %tmp12 = call i32  @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br i1 %tmp13, label %bb14, label %bb18
; CHECK: br i1 %tmp13,{{.*}}!prof [[WEIGHT]]

bb14:                                             ; preds = %bb10
  %tmp16 = call i32  @too()
  %tmp17 = icmp sgt i32 %arg2, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14, %bb10, %bb
  %tmp19 = phi i32 [ 6, %bb10 ], [ 6, %bb ], [ %tmp16, %bb14 ]
  %tmp20 = xor i32 %tmp19, 3
  %tmp23 = sext i32 %tmp20 to i64
  %tmp24 = call i64 @llvm.expect.i64(i64 %tmp23, i64 4)
  %tmp25 = icmp ne i64 %tmp24, 0
  br i1 %tmp25, label %bb26, label %bb28
; CHECK: br i1 %tmp25,{{.*}}!prof [[WEIGHT]]

bb26:                                             ; preds = %bb18
  %tmp27 = call i32 @goo()
  br label %bb30

bb28:                                             ; preds = %bb18
  %tmp29 = call i32 @hoo()
  br label %bb30
bb30:                                             ; preds = %bb28, %bb26
  ret void
}

define void @foo_i8_sext(i32 %arg, i32 %arg1, i8 %arg2, i32 %arg3)  {
; CHECK-LABEL: void @foo_i8_sext
bb:
  %tmp8 = call i32  @goo()
  %tmp9 = icmp sgt i32 %tmp8, %arg
  br i1 %tmp9, label %bb10, label %bb18
; CHECK: br i1 %tmp9,{{.*}}!prof [[WEIGHT]]

bb10:                                             ; preds = %bb
  %tmp12 = call i32  @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br i1 %tmp13, label %bb14, label %bb18
; CHECK: br i1 %tmp13,{{.*}}!prof [[WEIGHT]]

bb14:                                             ; preds = %bb10
  %tmp16 = call i8  @too8()
  %tmp17 = icmp sgt i8 %arg2, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14, %bb10, %bb
  %tmp19 = phi i8 [ 255, %bb10 ], [ 255, %bb ], [ %tmp16, %bb14 ]
  %tmp23 = sext i8 %tmp19 to i64
; after sign extension, the operand value becomes -1 which does not match 255
  %tmp24 = call i64 @llvm.expect.i64(i64 %tmp23, i64 255)
  %tmp25 = icmp ne i64 %tmp24, 0
  br i1 %tmp25, label %bb26, label %bb28
; CHECK: br i1 %tmp25,{{.*}}!prof [[WEIGHT]]

bb26:                                             ; preds = %bb18
  %tmp27 = call i32 @goo()
  br label %bb30

bb28:                                             ; preds = %bb18
  %tmp29 = call i32 @hoo()
  br label %bb30
bb30:                                             ; preds = %bb28, %bb26
  ret void
}

define void @foo_i8_sext_not_unlikely(i32 %arg, i32 %arg1, i8 %arg2, i32 %arg3)  {
; CHECK-LABEL: void @foo_i8_sext_not_unlikely
bb:
  %tmp8 = call i32  @goo()
  %tmp9 = icmp sgt i32 %tmp8, %arg
  br i1 %tmp9, label %bb10, label %bb18
; CHECK: br i1 %tmp9
; CHECK-NOT: !prof

bb10:                                             ; preds = %bb
  %tmp12 = call i32  @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br i1 %tmp13, label %bb14, label %bb18
; CHECK: br i1 %tmp13
; CHECK-NOT: !prof

bb14:                                             ; preds = %bb10
  %tmp16 = call i8  @too8()
  %tmp17 = icmp sgt i8 %arg2, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14, %bb10, %bb
  %tmp19 = phi i8 [ 255, %bb10 ], [ 255, %bb ], [ %tmp16, %bb14 ]
  %tmp23 = sext i8 %tmp19 to i64
; after sign extension, the operand value becomes -1 which matches -1
  %tmp24 = call i64 @llvm.expect.i64(i64 %tmp23, i64 -1)
  %tmp25 = icmp ne i64 %tmp24, 0
  br i1 %tmp25, label %bb26, label %bb28
; CHECK: br i1 %tmp25,{{.*}}!prof [[WEIGHT]]

bb26:                                             ; preds = %bb18
  %tmp27 = call i32 @goo()
  br label %bb30

bb28:                                             ; preds = %bb18
  %tmp29 = call i32 @hoo()
  br label %bb30
bb30:                                             ; preds = %bb28, %bb26
  ret void
}


define void @foo_i32_xor_not_unlikely(i32 %arg, i32 %arg1, i32 %arg2, i32 %arg3)  {
; CHECK-LABEL: void @foo_i32_xor_not_unlikely
bb:
  %tmp8 = call i32 @goo()
  %tmp9 = icmp sgt i32 %tmp8, %arg
  br i1 %tmp9, label %bb10, label %bb18
; CHECK: br i1 %tmp9
; CHECK-NOT: !prof

bb10:                                             ; preds = %bb
  %tmp12 = call i32  @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br i1 %tmp13, label %bb14, label %bb18
; CHECK: br i1 %tmp13
; CHECK-NOT: !prof

bb14:                                             ; preds = %bb10
  %tmp16 = call i32 @too()
  %tmp17 = icmp sgt i32 %arg2, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb14, %bb10, %bb
  %tmp19 = phi i32 [ 6, %bb10 ], [ 6, %bb ], [ %tmp16, %bb14 ]
  %tmp20 = xor i32 %tmp19, 2
  %tmp23 = sext i32 %tmp20 to i64
  %tmp24 = call i64 @llvm.expect.i64(i64 %tmp23, i64 4)
  %tmp25 = icmp ne i64 %tmp24, 0
  br i1 %tmp25, label %bb26, label %bb28
; CHECK: br i1 %tmp25,{{.*}}!prof [[WEIGHT]]

bb26:                                             ; preds = %bb18
  %tmp27 = call i32 @goo()
  br label %bb30

bb28:                                             ; preds = %bb18
  %tmp29 = call i32  @hoo()
  br label %bb30

bb30:                                             ; preds = %bb28, %bb26
  ret void
}

declare i32 @goo()

declare i32 @hoo()

declare i32 @too()

declare i8 @too8()

; Function Attrs: nounwind readnone
declare i64 @llvm.expect.i64(i64, i64)

!llvm.ident = !{!0}

!0 = !{!"clang version 5.0.0 (trunk 302965)"}
; CHECK: [[WEIGHT]] = !{!"branch_weights", !"expected", i32 2000, i32 1}
; CHECK: [[WEIGHT2]] = !{!"branch_weights", !"expected", i32 1, i32 2000}
