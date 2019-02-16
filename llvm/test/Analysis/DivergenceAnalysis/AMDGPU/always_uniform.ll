; RUN: opt  -mtriple amdgcn-unknown-amdhsa -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s

define amdgpu_kernel void @workitem_id_x() #1 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT:  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %first.lane = call i32 @llvm.amdgcn.readfirstlane(i32 %id.x)
; CHECK-NOT: DIVERGENT:  %first.lane = call i32 @llvm.amdgcn.readfirstlane(i32 %id.x)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.readfirstlane(i32) #0

attributes #0 = { nounwind readnone }
