; RUN: opt < %s -S -mtriple=amdgpu-- -passes=loop-unroll | FileCheck %s

; Check that the threshold used for partial unrolling is independent of the
; threshold used for full unrolling. A partial threshold of 30 produces two
; copies of the loop body while the threshold for full unrolling remains zero.
; CHECK-LABEL: @partial_unroll_threshold(
; CHECK: partial.body:
; CHECK: store i32
; CHECK: br i1
; CHECK: partial.body.1:
; CHECK: store i32
; CHECK-NOT: partial.body.2:
; CHECK: ret void

define void @partial_unroll_threshold(ptr addrspace(1) %a,
                                      ptr addrspace(1) %b) #0 {
entry:
  br label %partial.body

partial.body:                                     ; preds = %entry, %partial.body
  %iv = phi i64 [ 1, %entry ], [ %iv.next, %partial.body ]
  %src = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %iv
  %value = load i32, ptr addrspace(1) %src, align 4
  %index = sext i32 %value to i64
  %dst = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %index
  %stored = trunc i64 %iv to i32
  store i32 %stored, ptr addrspace(1) %dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 20
  br i1 %exitcond, label %exit, label %partial.body

exit:                                             ; preds = %partial.body
  ret void
}

attributes #0 = { "amdgpu-unroll-threshold"="0" "amdgpu-partial-unroll-threshold"="30" }
