; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK-LABEL: UniformityInfo for function 'test_waterfall_readlane':
; CHECK:     DIVERGENT:   %gep.in = getelementptr <2 x i32>, ptr addrspace(1) %in, i32 %tid
; CHECK:     DIVERGENT:   %args = load <2 x i32>, ptr addrspace(1) %gep.in, align 8
; CHECK:     DIVERGENT:   %value = extractelement <2 x i32> %args, i32 0
; CHECK:     DIVERGENT:   %lane = extractelement <2 x i32> %args, i32 1
; CHECK-NOT: DIVERGENT:   %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 0, i32 %lane)
; CHECK-NOT: DIVERGENT:   %readlane = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %lane)
; CHECK-NOT: DIVERGENT:   %readlane1 = call i32 @llvm.amdgcn.readlane(i32 %value, i32 %readlane)
; CHECK:     DIVERGENT:   %readlane2 = call i32 @llvm.amdgcn.waterfall.end.i32(i32 %wf_token, i32 %readlane1)
; CHECK:     DIVERGENT:   store i32 %readlane2, ptr addrspace(1) %out, align 4
define amdgpu_ps void @test_waterfall_readlane(i32 addrspace(1)* inreg %out, <2 x i32> addrspace(1)* inreg %in, i32 %tid) #1 {
  %gep.in = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 %tid
  %args = load <2 x i32>, <2 x i32> addrspace(1)* %gep.in
  %value = extractelement <2 x i32> %args, i32 0
  %lane = extractelement <2 x i32> %args, i32 1
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 0, i32 %lane)
  %readlane = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %lane)
  %readlane1 = call i32 @llvm.amdgcn.readlane(i32 %value, i32 %readlane)
  %readlane2 = call i32 @llvm.amdgcn.waterfall.end.i32(i32 %wf_token, i32 %readlane1)
  store i32 %readlane2, i32 addrspace(1)* %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.waterfall.begin.i32(i32, i32)
declare i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32, i32)
declare i32 @llvm.amdgcn.readlane(i32, i32)
declare i32 @llvm.amdgcn.waterfall.end.i32(i32, i32)
