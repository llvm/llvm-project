; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji -passes=loop-vectorize < %s | FileCheck %s


; For AMDGPU, loop unroll in loop vectorizer is disabled when VF==1.
;
; CHECK-LABEL: @small_loop(
; CHECK: store i32
; CHECK-NOT: store i32
; CHECK: ret
define amdgpu_kernel void @small_loop(i32* nocapture %inArray, i32 %size) nounwind {
entry:
  %0 = icmp sgt i32 %size, 0
  br i1 %0, label %loop, label %exit

loop:                                          ; preds = %entry, %loop
  %iv = phi i32 [ %iv1, %loop ], [ 0, %entry ]
  %1 = getelementptr inbounds i32, i32* %inArray, i32 %iv
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 6
  store i32 %3, i32* %1, align 4
  %iv1 = add i32 %iv, 1
;  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %cond = icmp eq i32 %iv1, %size
  br i1 %cond, label %exit, label %loop

exit:                                         ; preds = %loop, %entry
  ret void
}
