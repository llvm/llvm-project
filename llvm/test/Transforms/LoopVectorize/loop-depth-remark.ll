; RUN: opt -disable-output -passes=loop-vectorize -pass-remarks=loop-vectorize 2>&1 %s | FileCheck %s

; CHECK: remark: {{.*}} vectorized loop
; CHECK-SAME: VectorizationFactor
; CHECK-SAME: InterleaveCount
; CHECK-SAME: LoopDepth

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @add_arrays(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %n) {
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep_a = getelementptr inbounds float, ptr %a, i64 %iv
  %gep_b = getelementptr inbounds float, ptr %b, i64 %iv
  %gep_c = getelementptr inbounds float, ptr %c, i64 %iv
  %load_b = load float, ptr %gep_b, align 4
  %load_c = load float, ptr %gep_c, align 4
  %add = fadd float %load_b, %load_c
  store float %add, ptr %gep_a, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}
