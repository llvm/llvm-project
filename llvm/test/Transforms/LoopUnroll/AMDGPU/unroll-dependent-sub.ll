; RUN: opt -S -mtriple=amdgcn-- -passes=loop-unroll -debug-only=AMDGPUtti < %s 2>&1 | FileCheck %s

; For @dependent_sub_fullunroll, the threshold bonus should apply
; CHECK: due to subloop's trip count becoming runtime-independent after unrolling

; For @dependent_sub_no_fullunroll, the threshold bonus should not apply
; CHECK-NOT: due to subloop's trip count becoming runtime-independent after unrolling

; Check that the outer loop of a double-nested loop where the inner loop's trip
; count depends exclusively on constants and the outer IV is fully unrolled
; thanks to receiving a threshold bonus in AMDGPU's TTI.

; CHECK-LABEL: @dependent_sub_fullunroll
; CHECK: inner.header_latch_exiting.7
; CHECK: outer.latch_exiting.7

define void @dependent_sub_fullunroll(ptr noundef %mem) {
entry:
  br label %outer.header

outer.header:                                                 ; preds = %entry, %outer.latch_exiting
  %outer.iv = phi i32 [ 0, %entry ], [ %outer.iv_next, %outer.latch_exiting ]
  br label %inner.header_latch_exiting

inner.header_latch_exiting:                                   ; preds = %outer.header, %inner.header_latch_exiting
  %inner.iv = phi i32 [ %outer.iv, %outer.header ], [ %inner.iv_next, %inner.header_latch_exiting ]
  %inner.iv_next = add nuw nsw i32 %inner.iv, 1
  %outer.iv.ext = zext nneg i32 %outer.iv to i64
  %idx_part = mul nuw nsw i64 %outer.iv.ext, 16 
  %inner.iv.ext = zext nneg i32 %inner.iv to i64
  %idx = add nuw nsw i64 %idx_part, %inner.iv.ext 
  %addr = getelementptr inbounds i8, ptr %mem, i64 %idx
  store i32 0, ptr %addr
  %inner.cond = icmp ult i32 %inner.iv_next, 8
  br i1 %inner.cond, label %inner.header_latch_exiting, label %outer.latch_exiting, !llvm.loop !1

outer.latch_exiting:                                          ; preds = %inner.header_latch_exiting
  %outer.iv_next = add nuw nsw i32 %outer.iv, 1
  %outer.cond = icmp ult i32 %outer.iv_next, 8
  br i1 %outer.cond, label %outer.header, label %end, !llvm.loop !1
  
end:                                                          ; preds = %outer.latch_exiting
  ret void
}

; Check that the outer loop of the same loop nest as dependent_sub_fullunroll
; is not fully unrolled when the inner loop's final IV value depends on a
; function argument instead of a combination of the outer IV and constants.

; CHECK-LABEL: @dependent_sub_no_fullunroll
; CHECK-NOT: outer.latch_exiting.7
; CHECK-NOT: outer.latch_exiting.7

define void @dependent_sub_no_fullunroll(ptr noundef %mem, i32 noundef %inner.ub) {
entry:
  br label %outer.header

outer.header:                                                 ; preds = %entry, %outer.latch_exiting
  %outer.iv = phi i32 [ 0, %entry ], [ %outer.iv_next, %outer.latch_exiting ]
  br label %inner.header_latch_exiting

inner.header_latch_exiting:                                   ; preds = %outer.header, %inner.header_latch_exiting
  %inner.iv = phi i32 [ %outer.iv, %outer.header ], [ %inner.iv_next, %inner.header_latch_exiting ]
  %inner.iv_next = add nuw nsw i32 %inner.iv, 1
  %outer.iv.ext = zext nneg i32 %outer.iv to i64
  %idx_part = mul nuw nsw i64 %outer.iv.ext, 16 
  %inner.iv.ext = zext nneg i32 %inner.iv to i64
  %idx = add nuw nsw i64 %idx_part, %inner.iv.ext 
  %addr = getelementptr inbounds i8, ptr %mem, i64 %idx
  store i32 0, ptr %addr
  %inner.cond = icmp ult i32 %inner.iv_next, %inner.ub
  br i1 %inner.cond, label %inner.header_latch_exiting, label %outer.latch_exiting, !llvm.loop !1

outer.latch_exiting:                                          ; preds = %inner.header_latch_exiting
  %outer.iv_next = add nuw nsw i32 %outer.iv, 1
  %outer.cond = icmp ult i32 %outer.iv_next, 8
  br i1 %outer.cond, label %outer.header, label %end, !llvm.loop !1
  
end:                                                          ; preds = %outer.latch_exiting
  ret void
}

!1 = !{!1, !2}
!2 = !{!"amdgpu.loop.unroll.threshold", i32 100}
