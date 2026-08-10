; RUN: opt < %s -O3 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @no_vectorization(i32 %n, i32 %a, i32 %b) {
; CHECK-LABEL: no_vectorization(
; CHECK-NOT: <4 x i32>
; CHECK-NOT: <4 x i1>
entry:
  %cmp.5 = icmp sgt i32 %n, 0
  br i1 %cmp.5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %add = add nsw i32 %i.06, %a
  %mul = mul nsw i32 %add, %b
  %cmp1 = icmp sgt i32 %mul, -1
  tail call void @llvm.assume(i1 %cmp1)
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

declare void @llvm.assume(i1) #0

attributes #0 = { nounwind }

!nvvm.annotations = !{!0}
!0 = !{ptr @no_vectorization, !"kernel", i32 1}
