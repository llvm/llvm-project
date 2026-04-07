; RUN: opt -passes='print<scalar-evolution>,loop-unroll<upperbound>,print<scalar-evolution>' -unroll-max-upperbound=32 -disable-output %s 2>&1 | FileCheck %s

; Verify SCEV check all predecessors of a merge block (loop header of the
; second loop) to compute constant trip count after unrolling the first loop.

; Disable loop 2 unroll to check SCEV computation after loop 1 unroll.

; CHECK: Determining loop execution counts for: @test_merge_is_loop_header
; CHECK: Loop %for.body2: constant max backedge-taken count is i32 31
; CHECK: Loop %for.body: constant max backedge-taken count is i32 31

; CHECK: Determining loop execution counts for: @test_merge_is_loop_header
; CHECK: Loop %for.body2: constant max backedge-taken count is i32 31

declare i32 @llvm.umin.i32(i32, i32)

define void @test_merge_is_loop_header(i32 %x) {
entry:
  %results = alloca i64, align 8
  %trip.count = tail call i32 @llvm.umin.i32(i32 %x, i32 32)
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
  br i1 %cmp2, label %for.end, label %for.body2, !llvm.loop !0

for.body2:
  %i2 = phi i32 [ %inc2, %for.cond ], [ 0, %for.body ]
  %tmp = load i64, ptr %results, align 8
  br label %for.cond

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
