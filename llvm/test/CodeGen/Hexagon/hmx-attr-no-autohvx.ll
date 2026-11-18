; A "hexagon_hmx" function must stay scalar even with the HVX loop vectorizer
; enabled, so no HVX unit is acquired on the HMX thread.

; RUN: opt -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b -hexagon-autohvx \
; RUN:     -passes=loop-vectorize -S < %s | FileCheck %s

define void @hmx_stays_scalar(ptr %dst, ptr %a, ptr %b, i32 %n) #0 {
; CHECK-LABEL: @hmx_stays_scalar(
; CHECK-NOT:     <32 x i32>
; CHECK-NOT:     <64 x i32>
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %pa = getelementptr inbounds i32, ptr %a, i32 %iv
  %pb = getelementptr inbounds i32, ptr %b, i32 %iv
  %pd = getelementptr inbounds i32, ptr %dst, i32 %iv
  %va = load i32, ptr %pa, align 4
  %vb = load i32, ptr %pb, align 4
  %sum = add nsw i32 %va, %vb
  store i32 %sum, ptr %pd, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %done = icmp eq i32 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Same loop without the attribute is vectorized, so the check above is not
; passing for an unrelated reason.
define void @no_attr_vectorizes(ptr %dst, ptr %a, ptr %b, i32 %n) #1 {
; CHECK-LABEL: @no_attr_vectorizes(
; CHECK:         <32 x i32>
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %pa = getelementptr inbounds i32, ptr %a, i32 %iv
  %pb = getelementptr inbounds i32, ptr %b, i32 %iv
  %pd = getelementptr inbounds i32, ptr %dst, i32 %iv
  %va = load i32, ptr %pa, align 4
  %vb = load i32, ptr %pb, align 4
  %sum = add nsw i32 %va, %vb
  store i32 %sum, ptr %pd, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %done = icmp eq i32 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { nounwind "hexagon_hmx" "target-features"="+hvxv68,+hvx-length128b" }
attributes #1 = { nounwind "target-features"="+hvxv68,+hvx-length128b" }
