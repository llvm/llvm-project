; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<divergence>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: bb3:
; CHECK: DIVERGENT:       %Guard.bb4 = phi i1 [ true, %bb1 ], [ false, %bb2 ]
; CHECK: DIVERGENT:       br i1 %Guard.bb4, label %bb4, label %bb5

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.x() #0

define protected amdgpu_kernel void @test() {
bb0:
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %i5 = icmp eq i32 %tid.x, -1
  br label %bb1

bb1:                                              ; preds = %bb2, %bb0
  %lsr.iv = phi i32 [ 7, %bb0 ], [ %lsr.iv.next, %bb2 ]
  br i1 %i5, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  %lsr.iv.next = add nsw i32 %lsr.iv, -1
  %i14 = icmp eq i32 %lsr.iv.next, 0
  br i1 %i14, label %bb3, label %bb1

bb3:                                              ; preds = %bb2, %bb1
  %Guard.bb4 = phi i1 [ true, %bb1 ], [ false, %bb2 ]
  br i1 %Guard.bb4, label %bb4, label %bb5

bb4:                                              ; preds = %bb3
  br label %bb5

bb5:                                              ; preds = %bb3, %bb4
  ret void
}

attributes #0 = { nounwind readnone speculatable }
