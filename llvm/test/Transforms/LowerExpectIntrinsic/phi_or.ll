; RUN: opt -lower-expect  -S -o - < %s | FileCheck %s
; RUN: opt -S -passes='function(lower-expect)' < %s | FileCheck %s
; 
; if (__builtin_expect((x > goo() || y > hoo()), 1)) {
;  ..
; }
; For the above case, only the second branch should be
; annotated.
; if (__builtin_expect((x > goo() || y > hoo()), 0)) {
;  ..
; }
; For the above case, two branches should be annotated.
; Function Attrs: noinline nounwind uwtable
define void @foo(i32 %arg, i32 %arg1, i32 %arg2, i32 %arg3)  {
; CHECK-LABEL: void @foo
bb:
  %tmp8 = call i32 @goo()
  %tmp9 = icmp slt i32 %arg, %tmp8
  br i1 %tmp9, label %bb14, label %bb10
; CHECK: br i1 %tmp9
; CHECK-NOT: br i1 %tmp9{{.*}}!prof

bb10:                                             ; preds = %bb
  %tmp12 = call i32  @hoo()
  %tmp13 = icmp sgt i32 %arg1, %tmp12
  br label %bb14

bb14:                                             ; preds = %bb10, %bb
  %tmp15 = phi i1 [ true, %bb ], [ %tmp13, %bb10 ]
  %tmp16 = zext i1 %tmp15 to i32
  %tmp17 = sext i32 %tmp16 to i64
  %expect = call i64 @llvm.expect.i64(i64 %tmp17, i64 1)
  %tmp18 = icmp ne i64 %expect, 0
  br i1 %tmp18, label %bb19, label %bb21
; CHECK: br i1 %tmp18{{.*}}!prof [[WEIGHT:![0-9]+]]

bb19:                                             ; preds = %bb14
  %tmp20 = call i32 @goo()
  br label %bb23

bb21:                                             ; preds = %bb14
  %tmp22 = call i32  @hoo()
  br label %bb23

bb23:                                             ; preds = %bb21, %bb19
  ret void
}

define void @foo2(i32 %arg, i32 %arg1, i32 %arg2, i32 %arg3)  {
; CHECK-LABEL: void @foo2
bb:
  %tmp = alloca i32, align 4
  %tmp4 = alloca i32, align 4
  %tmp5 = alloca i32, align 4
  %tmp6 = alloca i32, align 4
  store i32 %arg, i32* %tmp, align 4
  store i32 %arg1, i32* %tmp4, align 4
  store i32 %arg2, i32* %tmp5, align 4
  store i32 %arg3, i32* %tmp6, align 4
  %tmp7 = load i32, i32* %tmp, align 4
  %tmp8 = call i32  @goo()
  %tmp9 = icmp slt i32 %tmp7, %tmp8
  br i1 %tmp9, label %bb14, label %bb10
; CHECK: br i1 %tmp9{{.*}}!prof [[WEIGHT2:![0-9]+]]

bb10:                                             ; preds = %bb
  %tmp11 = load i32, i32* %tmp5, align 4
  %tmp12 = call i32 @hoo()
  %tmp13 = icmp sgt i32 %tmp11, %tmp12
  br label %bb14

bb14:                                             ; preds = %bb10, %bb
  %tmp15 = phi i1 [ true, %bb ], [ %tmp13, %bb10 ]
  %tmp16 = zext i1 %tmp15 to i32
  %tmp17 = sext i32 %tmp16 to i64
  %expect = call i64 @llvm.expect.i64(i64 %tmp17, i64 0)
  %tmp18 = icmp ne i64 %expect, 0
  br i1 %tmp18, label %bb19, label %bb21
; CHECK: br i1 %tmp18{{.*}}!prof [[WEIGHT2]]

bb19:                                             ; preds = %bb14
  %tmp20 = call i32 @goo()
  br label %bb23

bb21:                                             ; preds = %bb14
  %tmp22 = call i32 @hoo()
  br label %bb23

bb23:                                             ; preds = %bb21, %bb19
  ret void
}

declare i32 @goo() 
declare i32 @hoo() 
declare i64 @llvm.expect.i64(i64, i64) 


!llvm.ident = !{!0}


!0 = !{!"clang version 5.0.0 (trunk 302965)"}
; CHECK: [[WEIGHT]] = !{!"branch_weights", i32 2000, i32 1}
; CHECK: [[WEIGHT2]] = !{!"branch_weights", i32 1, i32 2000}
