; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope %s

declare i32 @llvm.amdgcn.readlane(i32, i32) #0

; CHECK-LABEL: {{^}}test_readlane_sreg_sreg:
; CHECK-NOT: v_readlane_b32
define amdgpu_kernel void @test_readlane_sreg_sreg(i32 %src0, i32 %src1) #1 {
  %readlane = call i32 @llvm.amdgcn.readlane(i32 %src0, i32 %src1)
  call void asm sideeffect "; use $0", "s"(i32 %readlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vreg_sreg:
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_readlane_vreg_sreg(i32 %src0, i32 %src1) #1 {
  %vgpr = call i32 asm sideeffect "; def $0", "=v"()
  %readlane = call i32 @llvm.amdgcn.readlane(i32 %vgpr, i32 %src1)
  call void asm sideeffect "; use $0", "s"(i32 %readlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_imm_sreg:
; CHECK-NOT: v_readlane_b32
define amdgpu_kernel void @test_readlane_imm_sreg(ptr addrspace(1) %out, i32 %src1) #1 {
  %readlane = call i32 @llvm.amdgcn.readlane(i32 32, i32 %src1)
  store i32 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vregs:
; CHECK: v_readfirstlane_b32 [[LANE:s[0-9]+]], v{{[0-9]+}}
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, [[LANE]]
define amdgpu_kernel void @test_readlane_vregs(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr <2 x i32>, ptr addrspace(1) %in, i32 %tid
  %args = load <2 x i32>, ptr addrspace(1) %gep.in
  %value = extractelement <2 x i32> %args, i32 0
  %lane = extractelement <2 x i32> %args, i32 1
  %readlane = call i32 @llvm.amdgcn.readlane(i32 %value, i32 %lane)
  store i32 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: m0 should be folded.
; CHECK-LABEL: {{^}}test_readlane_m0_sreg:
; CHECK: s_mov_b32 m0, -1
; CHECK: v_mov_b32_e32 [[VVAL:v[0-9]]], m0
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VVAL]]
define amdgpu_kernel void @test_readlane_m0_sreg(ptr addrspace(1) %out, i32 %src1) #1 {
  %m0 = call i32 asm "s_mov_b32 m0, -1", "={m0}"()
  %readlane = call i32 @llvm.amdgcn.readlane(i32 %m0, i32 %src1)
  store i32 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vgpr_imm:
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, 32
define amdgpu_kernel void @test_readlane_vgpr_imm(ptr addrspace(1) %out) #1 {
  %vgpr = call i32 asm sideeffect "; def $0", "=v"()
  %readlane = call i32 @llvm.amdgcn.readlane(i32 %vgpr, i32 32) #0
  store i32 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_copy_from_sgpr:
; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_mov_b32 [[SGPR:s[0-9]+]]
; CHECK: ;;#ASMEND
; CHECK-NOT: [[SGPR]]
; CHECK-NOT: readlane
; CHECK: v_mov_b32_e32 [[VCOPY:v[0-9]+]], [[SGPR]]
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VCOPY]]
define amdgpu_kernel void @test_readlane_copy_from_sgpr(ptr addrspace(1) %out) #1 {
  %sgpr = call i32 asm "s_mov_b32 $0, 0", "=s"()
  %readfirstlane = call i32 @llvm.amdgcn.readlane(i32 %sgpr, i32 7)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }
