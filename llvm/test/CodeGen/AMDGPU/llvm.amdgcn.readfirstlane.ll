; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.amdgcn.readfirstlane(i32) #0

; CHECK-LABEL: {{^}}test_readfirstlane:
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_readfirstlane(i32 addrspace(1)* %out, i32 %src) #1 {
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 %src)
  store i32 %readfirstlane, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_imm:
; CHECK: v_mov_b32_e32 [[VVAL:v[0-9]]], 32
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, [[VVAL]]
define amdgpu_kernel void @test_readfirstlane_imm(i32 addrspace(1)* %out) #1 {
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 32)
  store i32 %readfirstlane, i32 addrspace(1)* %out, align 4
  ret void
}

; TODO: m0 should be folded.
; CHECK-LABEL: {{^}}test_readfirstlane_m0:
; CHECK: s_mov_b32 m0, -1
; CHECK: s_mov_b32 [[COPY_M0:s[0-9]+]], m0
; CHECK: v_mov_b32_e32 [[VVAL:v[0-9]]], [[COPY_M0]]
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, [[VVAL]]
define amdgpu_kernel void @test_readfirstlane_m0(i32 addrspace(1)* %out) #1 {
  %m0 = call i32 asm "s_mov_b32 m0, -1", "={M0}"()
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 %m0)
  store i32 %readfirstlane, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
