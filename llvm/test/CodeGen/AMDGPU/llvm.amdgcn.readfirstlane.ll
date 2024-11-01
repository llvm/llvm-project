; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope %s

declare i32 @llvm.amdgcn.readfirstlane(i32) #0

; CHECK-LABEL: {{^}}test_readfirstlane:
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, v2
define void @test_readfirstlane(ptr addrspace(1) %out, i32 %src) #1 {
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 %src)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_imm:
; CHECK: s_mov_b32 [[SGPR_VAL:s[0-9]]], 32
; CHECK-NOT: [[SGPR_VAL]]
; CHECK: ; use [[SGPR_VAL]]
define amdgpu_kernel void @test_readfirstlane_imm(ptr addrspace(1) %out) #1 {
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 32)
  call void asm sideeffect "; use $0", "s"(i32 %readfirstlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_imm_fold:
; CHECK: v_mov_b32_e32 [[VVAL:v[0-9]]], 32
; CHECK-NOT: [[VVAL]]
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VVAL]]
define amdgpu_kernel void @test_readfirstlane_imm_fold(ptr addrspace(1) %out) #1 {
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 32)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_m0:
; CHECK: s_mov_b32 m0, -1
; CHECK: v_mov_b32_e32 [[VVAL:v[0-9]]], m0
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VVAL]]
define amdgpu_kernel void @test_readfirstlane_m0(ptr addrspace(1) %out) #1 {
  %m0 = call i32 asm "s_mov_b32 m0, -1", "={m0}"()
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 %m0)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_copy_from_sgpr:
; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_mov_b32 [[SGPR:s[0-9]+]]
; CHECK: ;;#ASMEND
; CHECK-NOT: [[SGPR]]
; CHECK-NOT: readfirstlane
; CHECK: v_mov_b32_e32 [[VCOPY:v[0-9]+]], [[SGPR]]
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VCOPY]]
define amdgpu_kernel void @test_readfirstlane_copy_from_sgpr(ptr addrspace(1) %out) #1 {
  %sgpr = call i32 asm "s_mov_b32 $0, 0", "=s"()
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 %sgpr)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; Make sure this doesn't crash.
; CHECK-LABEL: {{^}}test_readfirstlane_fi:
; CHECK: s_mov_b32 [[FIVAL:s[0-9]]], 4
define amdgpu_kernel void @test_readfirstlane_fi(ptr addrspace(1) %out) #1 {
  %alloca = alloca i32, addrspace(5)
  %int = ptrtoint ptr addrspace(5) %alloca to i32
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 %int)
  call void asm sideeffect "; use $0", "s"(i32 %readfirstlane)
  ret void
}

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
