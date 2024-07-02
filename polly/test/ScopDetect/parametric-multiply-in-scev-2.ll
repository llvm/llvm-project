; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s


; CHECK-NOT: Valid Region
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @blam(ptr %A, ptr %B) {
bb:
  %tmp1 = alloca i64
  %tmp2 = shl i64 2, undef
  %tmp3 = shl i64 2, undef
  %tmp4 = mul nsw i64 %tmp2, %tmp3
  br label %loop

loop:
  %indvar = phi i64 [ %indvar.next, %loop ], [ 0, %bb ]
  %gep = getelementptr inbounds i64, ptr %tmp1, i64 %indvar
  %tmp12 = load i64, ptr %gep
  %tmp13 = mul nsw i64 %tmp12, %tmp4
  %ptr = getelementptr inbounds float, ptr %B, i64 %tmp13
  %val = load float, ptr %ptr
  store float %val, ptr %A
  %indvar.next = add nsw i64 %indvar, 1
  br i1 false, label %loop, label %bb21

bb21:
  ret void
}
