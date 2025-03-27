; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mcpu=neoverse-v1 -disable-output %s -debug \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue 2>&1 | FileCheck %s

target triple="aarch64--linux-gnu"

; CHECK: LV: Checking a loop in 'gather_nxv4i32_loaded_index'
; CHECK: LV: Found an estimated cost of 81 for VF vscale x 4 For instruction:   %1 = load float, ptr %arrayidx3, align 4
define void @gather_nxv4i32_loaded_index(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %b, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds float, ptr %a, i64 %0
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %c, i64 %indvars.iv
  store float %1, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

; CHECK: LV: Checking a loop in 'scatter_nxv4i32_loaded_index'
; CHECK: LV: Found an estimated cost of 81 for VF vscale x 4 For instruction:   store float %1, ptr %arrayidx5, align 4
define void @scatter_nxv4i32_loaded_index(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %b, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %c, i64 %0
  store float %1, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

; NOTE: For runtime-determined strides the vectoriser versions the loop and adds SCEV checks
; to ensure the stride value is always 1. Therefore, it can assume a contiguous load and a cost of 1.
; CHECK: LV: Checking a loop in 'gather_nxv4i32_unknown_stride'
; CHECK: LV: Found an estimated cost of 1 for VF vscale x 4 For instruction:   %0 = load float, ptr %arrayidx, align 4
define void @gather_nxv4i32_unknown_stride(ptr noalias nocapture readonly %a, ptr noalias nocapture %b, i64 %stride, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %indvars.iv.stride2 = mul i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv.stride2
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  store float %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

; NOTE: For runtime-determined strides the vectoriser versions the loop and adds SCEV checks
; to ensure the stride value is always 1. Therefore, it can assume a contiguous load and cost is 1.
; CHECK: LV: Checking a loop in 'scatter_nxv4i32_unknown_stride'
; CHECK: LV: Found an estimated cost of 1 for VF vscale x 4 For instruction:   store float %0, ptr %arrayidx2, align 4
define void @scatter_nxv4i32_unknown_stride(ptr noalias nocapture readonly %a, ptr noalias nocapture %b, i64 %stride, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %indvars.iv.stride2 = mul i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %indvars.iv.stride2
  store float %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

; CHECK: LV: Checking a loop in 'gather_nxv4i32_stride2'
; CHECK: LV: Found an estimated cost of 2 for VF vscale x 4 For instruction:   %0 = load float, ptr %arrayidx, align 4
define void @gather_nxv4i32_stride2(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %indvars.iv.stride2 = mul i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv.stride2
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  store float %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

; CHECK: LV: Checking a loop in 'scatter_nxv4i32_stride2'
; CHECK: LV: Found an estimated cost of 81 for VF vscale x 4 For instruction:   store float %0, ptr %arrayidx2, align 4
define void @scatter_nxv4i32_stride2(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %indvars.iv.stride2 = mul i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %indvars.iv.stride2
  store float %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}


; CHECK: LV: Checking a loop in 'gather_nxv4i32_stride64'
; CHECK: LV: Found an estimated cost of 81 for VF vscale x 4 For instruction:   %0 = load float, ptr %arrayidx, align 4
define void @gather_nxv4i32_stride64(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %indvars.iv.stride2 = mul i64 %indvars.iv, 64
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv.stride2
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  store float %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

; CHECK: LV: Checking a loop in 'scatter_nxv4i32_stride64'
; CHECK: LV: Found an estimated cost of 81 for VF vscale x 4 For instruction:   store float %0, ptr %arrayidx2, align 4
define void @scatter_nxv4i32_stride64(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %indvars.iv.stride2 = mul i64 %indvars.iv, 64
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %indvars.iv.stride2
  store float %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}


attributes #0 = { vscale_range(1, 16) "target-features"="+sve" }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 1}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
