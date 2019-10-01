; RUN: opt  -mtriple amdgcn-unknown-amdhsa -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s

; CHECK: for function 'readfirstlane':
define amdgpu_kernel void @readfirstlane() {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT:  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %first.lane = call i32 @llvm.amdgcn.readfirstlane(i32 %id.x)
; CHECK-NOT: DIVERGENT:  %first.lane = call i32 @llvm.amdgcn.readfirstlane(i32 %id.x)
  ret void
}

; CHECK: for function 'icmp':
define amdgpu_kernel void @icmp(i32 inreg %x) {
; CHECK-NOT: DIVERGENT:  %icmp = call i64 @llvm.amdgcn.icmp.i32
  %icmp = call i64 @llvm.amdgcn.icmp.i32(i32 %x, i32 0, i32 33)
  ret void
}

; CHECK: for function 'fcmp':
define amdgpu_kernel void @fcmp(float inreg %x, float inreg %y) {
; CHECK-NOT: DIVERGENT:  %fcmp = call i64 @llvm.amdgcn.fcmp.i32
  %fcmp = call i64 @llvm.amdgcn.fcmp.i32(float %x, float %y, i32 33)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.readfirstlane(i32) #0
declare i64 @llvm.amdgcn.icmp.i32(i32, i32, i32) #1
declare i64 @llvm.amdgcn.fcmp.i32(float, float, i32) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone convergent }
