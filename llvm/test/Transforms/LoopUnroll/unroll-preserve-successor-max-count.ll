; RUN: opt -passes='loop-unroll<upperbound>' -S < %s | FileCheck %s

; Test that unrolling the first loop preserves the max trip count of the sibling
; successor loop via metadata, preventing SCEV info loss.

; CHECK-LABEL: @test_upperbound(
; CHECK-COUNT-4: store i64 0, ptr %results
; CHECK-COUNT-4: load i64, ptr %results

declare i32 @llvm.umin.i32(i32, i32)

declare void @use(i64)

define void @test_upperbound(i32 %x) {
entry:
  %results = alloca i64, align 8
  %trip.count = tail call i32 @llvm.umin.i32(i32 %x, i32 4)
  %cmp0 = icmp eq i32 %trip.count, 0
  br i1 %cmp0, label %for.end, label %for.body

for.body:
  %i = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  store i64 0, ptr %results, align 8
  %inc = add i32 %i, 1
  %cmp = icmp eq i32 %inc, %trip.count
  br i1 %cmp, label %for.body2, label %for.body

for.cond:
  %inc2 = add nuw nsw i32 %i2, 1
  %cmp2 = icmp eq i32 %inc2, %trip.count
  br i1 %cmp2, label %for.end, label %for.body2

for.body2:
  %i2 = phi i32 [ %inc2, %for.cond ], [ 0, %for.body ]
  %tmp = load i64, ptr %results, align 8
  tail call void @use(i64 %tmp)
  br label %for.cond

for.end:
  ret void
}
