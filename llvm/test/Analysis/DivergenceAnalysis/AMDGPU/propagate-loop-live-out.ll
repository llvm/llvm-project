; RUN: opt -mtriple amdgcn-unknown-amdhsa -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s

; CHECK: bb6:
; CHECK: DIVERGENT:       %.126.i355.i = phi i1 [ false, %bb5 ], [ true, %bb4 ]
; CHECK: DIVERGENT:       br i1 %.126.i355.i, label %bb7, label %bb8

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.x() #0

define protected amdgpu_kernel void @_Z23krnl_GPUITSFitterKerneli() {
bb0:
  %i4 = call i32 @llvm.amdgcn.workitem.id.x()
  %i5 = icmp eq i32 %i4, -1
  br label %bb1

bb1:                                              ; preds = %bb3, %bb0
  %lsr.iv = phi i32 [ %i1, %bb3 ], [ 7, %bb0 ]
  br i1 %i5, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  %lsr.iv.next = add nsw i32 %lsr.iv, -1
  %i14 = icmp eq i32 %lsr.iv.next, 0
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  %i1 = phi i32 [ %lsr.iv.next, %bb2 ], [ 0, %bb1 ]
  %i2 = phi i1 [ false, %bb2 ], [ true, %bb1 ]
  %i3 = phi i1 [ %i14, %bb2 ], [ true, %bb1 ]
  br i1 %i3, label %bb4, label %bb1

bb4:                                              ; preds = %bb3
  br i1 %i2, label %bb5, label %bb6

bb5:                                              ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb5, %bb4
  %.126.i355.i = phi i1 [ false, %bb5 ], [ true, %bb4 ]
  br i1 %.126.i355.i, label %bb7, label %bb8

bb7:                                              ; preds = %bb6
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  ret void
}

attributes #0 = { nounwind readnone speculatable }
