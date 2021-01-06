; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,CIGFX9 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx802 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,CIGFX9 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,GFX10 %s

declare i32 @llvm.amdgcn.writelane(i32, i32, i32) #0

; CHECK-LABEL: {{^}}test_writelane_sreg:
; CIGFX9: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_sreg(i32 addrspace(1)* %out, i32 %src0, i32 %src1) #1 {
  %oldval = load i32, i32 addrspace(1)* %out
  %writelane = call i32 @llvm.amdgcn.writelane(i32 %src0, i32 %src1, i32 %oldval)
  store i32 %writelane, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_sreg:
; CHECK: v_writelane_b32 v{{[0-9]+}}, 32, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_imm_sreg(i32 addrspace(1)* %out, i32 %src1) #1 {
  %oldval = load i32, i32 addrspace(1)* %out
  %writelane = call i32 @llvm.amdgcn.writelane(i32 32, i32 %src1, i32 %oldval)
  store i32 %writelane, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_vreg_lane:
; CHECK: v_readfirstlane_b32 [[LANE:s[0-9]+]], v{{[0-9]+}}
; CHECK: v_writelane_b32 v{{[0-9]+}}, 12, [[LANE]]
define amdgpu_kernel void @test_writelane_vreg_lane(i32 addrspace(1)* %out, <2 x i32> addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 %tid
  %args = load <2 x i32>, <2 x i32> addrspace(1)* %gep.in
  %oldval = load i32, i32 addrspace(1)* %out
  %lane = extractelement <2 x i32> %args, i32 1
  %writelane = call i32 @llvm.amdgcn.writelane(i32 12, i32 %lane, i32 %oldval)
  store i32 %writelane, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_m0_sreg:
; CHECK: s_mov_b32 m0, -1
; CIGFX9: s_mov_b32 [[COPY_M0:s[0-9]+]], m0
; CIGFX9: v_writelane_b32 v{{[0-9]+}}, [[COPY_M0]], m0
; GFX10: v_writelane_b32 v{{[0-9]+}}, m0, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_m0_sreg(i32 addrspace(1)* %out, i32 %src1) #1 {
  %oldval = load i32, i32 addrspace(1)* %out
  %m0 = call i32 asm "s_mov_b32 m0, -1", "={m0}"()
  %writelane = call i32 @llvm.amdgcn.writelane(i32 %m0, i32 %src1, i32 %oldval)
  store i32 %writelane, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm:
; CHECK: v_writelane_b32 v{{[0-9]+}}, s{{[0-9]+}}, 32
define amdgpu_kernel void @test_writelane_imm(i32 addrspace(1)* %out, i32 %src0) #1 {
  %oldval = load i32, i32 addrspace(1)* %out
  %writelane = call i32 @llvm.amdgcn.writelane(i32 %src0, i32 32, i32 %oldval) #0
  store i32 %writelane, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_sreg_oldval:
; CHECK: v_mov_b32_e32 [[OLDVAL:v[0-9]+]], s{{[0-9]+}}
; CIGFX9: v_writelane_b32 [[OLDVAL]], s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 [[OLDVAL]], s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_sreg_oldval(i32 inreg %oldval, i32 addrspace(1)* %out, i32 %src0, i32 %src1) #1 {
  %writelane = call i32 @llvm.amdgcn.writelane(i32 %src0, i32 %src1, i32 %oldval)
  store i32 %writelane, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_writelane_imm_oldval:
; CHECK: v_mov_b32_e32 [[OLDVAL:v[0-9]+]], 42
; CIGFX9: v_writelane_b32 [[OLDVAL]], s{{[0-9]+}}, m0
; GFX10: v_writelane_b32 [[OLDVAL]], s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_writelane_imm_oldval(i32 addrspace(1)* %out, i32 %src0, i32 %src1) #1 {
  %writelane = call i32 @llvm.amdgcn.writelane(i32 %src0, i32 %src1, i32 42)
  store i32 %writelane, i32 addrspace(1)* %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }
