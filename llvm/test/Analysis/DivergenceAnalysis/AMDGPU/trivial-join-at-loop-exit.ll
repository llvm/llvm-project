; RUN: opt -mtriple amdgcn-unknown-amdhsa -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s

; CHECK: bb2:
; CHECK-NOT: DIVERGENT:       %Guard.bb2 = phi i1 [ true, %bb1 ], [ false, %bb0 ]

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.x() #0

define protected amdgpu_kernel void @test2(i1 %uni) {
bb0:
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %i5 = icmp eq i32 %tid.x, -1
  br i1 %uni, label %bb1, label %bb2

bb1:                                              ; preds = %bb2, %bb0
  %lsr.iv = phi i32 [ 7, %bb0 ], [ %lsr.iv.next, %bb1 ]
  %lsr.iv.next = add nsw i32 %lsr.iv, -1
  br i1 %i5, label %bb2, label %bb1

bb2:                                              ; preds = %bb2, %bb1
  %Guard.bb2 = phi i1 [ true, %bb1 ], [ false, %bb0 ]
  ret void
}

attributes #0 = { nounwind readnone speculatable }
