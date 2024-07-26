; REQUIRES: asserts
; RUN: not --crash opt -p loop-vectorize -mtriple=s390x-unknown-linux -mcpu=z16 %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"

@src = external global [8 x i32], align 4

; Test case where scalar steps are used by both a VPReplicateRecipe (demands
; all scalar lanes) and a VPInstruction that only demands the first lane.
; Test case for https://github.com/llvm/llvm-project/issues/88849.
define void @test_scalar_iv_steps_used_by_replicate_and_first_lane_only_vpinst(ptr noalias %dst, ptr noalias %src.1) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %mul.iv = mul nsw i64 %iv, 4
  %gep.src.1 = getelementptr inbounds i8, ptr %src.1, i64 %mul.iv
  %l.1 = load i8, ptr %gep.src.1, align 1
  %c = icmp eq i8 %l.1, 0
  br i1 %c, label %then, label %loop.latch

then:
  %iv.or = or disjoint i64 %iv, 4
  %gep.src = getelementptr inbounds [8 x i32], ptr @src, i64 0, i64 %iv.or
  %l.2 = load i32, ptr %gep.src, align 4
  store i32 %l.2, ptr %dst, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 4
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}
