; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope %s

declare i32 @llvm.amdgcn.readlane.i32(i32, i32) #0
declare i64 @llvm.amdgcn.readlane.i64(i64, i32) #0
declare double @llvm.amdgcn.readlane.f64(double, i32) #0

; CHECK-LABEL: {{^}}test_readlane_sreg_sreg_i32:
; CHECK-NOT: v_readlane_b32
define amdgpu_kernel void @test_readlane_sreg_sreg_i32(i32 %src0, i32 %src1) #1 {
  %readlane = call i32 @llvm.amdgcn.readlane.i32(i32 %src0, i32 %src1)
  call void asm sideeffect "; use $0", "s"(i32 %readlane)
  ret void
}

; TODO: should optimize this as for i32
; CHECK-LABEL: {{^}}test_readlane_sreg_sreg_i64:
; CHECK: v_mov_b32_e32 [[VREG0:v[0-9]+]], {{s[0-9]+}}
; CHECK: v_mov_b32_e32 [[VREG1:v[0-9]+]], {{s[0-9]+}}
; CHECK: v_readlane_b32 {{s[0-9]+}}, [[VREG0]], {{s[0-9]+}}
; CHECK: v_readlane_b32 {{s[0-9]+}}, [[VREG1]], {{s[0-9]+}}
define amdgpu_kernel void @test_readlane_sreg_sreg_i64(i64 %src0, i32 %src1) #1 {
  %readlane = call i64 @llvm.amdgcn.readlane.i64(i64 %src0, i32 %src1)
  call void asm sideeffect "; use $0", "s"(i64 %readlane)
  ret void
}

; TODO: should optimize this as for i32
; CHECK-LABEL: {{^}}test_readlane_sreg_sreg_f64:
; CHECK: v_mov_b32_e32 [[VREG0:v[0-9]+]], {{s[0-9]+}}
; CHECK: v_mov_b32_e32 [[VREG1:v[0-9]+]], {{s[0-9]+}}
; CHECK: v_readlane_b32 {{s[0-9]+}}, [[VREG0]], {{s[0-9]+}}
; CHECK: v_readlane_b32 {{s[0-9]+}}, [[VREG1]], {{s[0-9]+}}
define amdgpu_kernel void @test_readlane_sreg_sreg_f64(double %src0, i32 %src1) #1 {
  %readlane = call double @llvm.amdgcn.readlane.f64(double %src0, i32 %src1)
  call void asm sideeffect "; use $0", "s"(double %readlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vreg_sreg_i32:
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_readlane_vreg_sreg_i32(i32 %src0, i32 %src1) #1 {
  %vgpr = call i32 asm sideeffect "; def $0", "=v"()
  %readlane = call i32 @llvm.amdgcn.readlane.i32(i32 %vgpr, i32 %src1)
  call void asm sideeffect "; use $0", "s"(i32 %readlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vreg_sreg_i64:
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_readlane_vreg_sreg_i64(i64 %src0, i32 %src1) #1 {
  %vgpr = call i64 asm sideeffect "; def $0", "=v"()
  %readlane = call i64 @llvm.amdgcn.readlane.i64(i64 %vgpr, i32 %src1)
  call void asm sideeffect "; use $0", "s"(i64 %readlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vreg_sreg_f64:
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @test_readlane_vreg_sreg_f64(double %src0, i32 %src1) #1 {
  %vgpr = call double asm sideeffect "; def $0", "=v"()
  %readlane = call double @llvm.amdgcn.readlane.f64(double %vgpr, i32 %src1)
  call void asm sideeffect "; use $0", "s"(double %readlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_imm_sreg_i32:
; CHECK-NOT: v_readlane_b32
define amdgpu_kernel void @test_readlane_imm_sreg_i32(ptr addrspace(1) %out, i32 %src1) #1 {
  %readlane = call i32 @llvm.amdgcn.readlane.i32(i32 32, i32 %src1)
  store i32 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_imm_sreg_i64:
; CHECK-NOT: v_readlane_b32
define amdgpu_kernel void @test_readlane_imm_sreg_i64(ptr addrspace(1) %out, i32 %src1) #1 {
  %readlane = call i64 @llvm.amdgcn.readlane.i64(i64 32, i32 %src1)
  store i64 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_imm_sreg_f64:
; CHECK-NOT: v_readlane_b32
define amdgpu_kernel void @test_readlane_imm_sreg_f64(ptr addrspace(1) %out, i32 %src1) #1 {
  %readlane = call double @llvm.amdgcn.readlane.f64(double 32.0, i32 %src1)
  store double %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vregs_i32:
; CHECK: v_readfirstlane_b32 [[LANE:s[0-9]+]], v{{[0-9]+}}
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, [[LANE]]
define amdgpu_kernel void @test_readlane_vregs_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr <2 x i32>, ptr addrspace(1) %in, i32 %tid
  %args = load <2 x i32>, ptr addrspace(1) %gep.in
  %value = extractelement <2 x i32> %args, i32 0
  %lane = extractelement <2 x i32> %args, i32 1
  %readlane = call i32 @llvm.amdgcn.readlane.i32(i32 %value, i32 %lane)
  store i32 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vregs_i64:
; CHECK: v_readfirstlane_b32 [[LANE:s[0-9]+]], v{{[0-9]+}}
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, [[LANE]]
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, [[LANE]]
define amdgpu_kernel void @test_readlane_vregs_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr <2 x i64>, ptr addrspace(1) %in, i32 %tid
  %args = load <2 x i64>, ptr addrspace(1) %gep.in
  %value = extractelement <2 x i64> %args, i32 0
  %lane = extractelement <2 x i64> %args, i32 1
  %lane32 = trunc i64 %lane to i32
  %readlane = call i64 @llvm.amdgcn.readlane.i64(i64 %value, i32 %lane32)
  store i64 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vregs_f64:
; CHECK: v_readfirstlane_b32 [[LANE:s[0-9]+]], v{{[0-9]+}}
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, [[LANE]]
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, [[LANE]]
define amdgpu_kernel void @test_readlane_vregs_f64(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr <2 x double>, ptr addrspace(1) %in, i32 %tid
  %args = load <2 x double>, ptr addrspace(1) %gep.in
  %value = extractelement <2 x double> %args, i32 0
  %lane = extractelement <2 x double> %args, i32 1
  %lane_cast = bitcast double %lane to i64
  %lane32 = trunc i64 %lane_cast to i32
  %readlane = call double @llvm.amdgcn.readlane.f64(double %value, i32 %lane32)
  store double %readlane, ptr addrspace(1) %out, align 4
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

; CHECK-LABEL: {{^}}test_readlane_vgpr_imm_i32:
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, 32
define amdgpu_kernel void @test_readlane_vgpr_imm_i32(ptr addrspace(1) %out) #1 {
  %vgpr = call i32 asm sideeffect "; def $0", "=v"()
  %readlane = call i32 @llvm.amdgcn.readlane.i32(i32 %vgpr, i32 32) #0
  store i32 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vgpr_imm_i64:
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, 32
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, 32
define amdgpu_kernel void @test_readlane_vgpr_imm_i64(ptr addrspace(1) %out) #1 {
  %vgpr = call i64 asm sideeffect "; def $0", "=v"()
  %readlane = call i64 @llvm.amdgcn.readlane.i64(i64 %vgpr, i32 32) #0
  store i64 %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_vgpr_imm_f64:
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, 32
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]+}}, 32
define amdgpu_kernel void @test_readlane_vgpr_imm_f64(ptr addrspace(1) %out) #1 {
  %vgpr = call double asm sideeffect "; def $0", "=v"()
  %readlane = call double @llvm.amdgcn.readlane.f64(double %vgpr, i32 32) #0
  store double %readlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readlane_copy_from_sgpr_i32:
; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_mov_b32 [[SGPR:s[0-9]+]]
; CHECK: ;;#ASMEND
; CHECK-NOT: [[SGPR]]
; CHECK-NOT: readlane
; CHECK: v_mov_b32_e32 [[VCOPY:v[0-9]+]], [[SGPR]]
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VCOPY]]
define amdgpu_kernel void @test_readlane_copy_from_sgpr_i32(ptr addrspace(1) %out) #1 {
  %sgpr = call i32 asm "s_mov_b32 $0, 0", "=s"()
  %readfirstlane = call i32 @llvm.amdgcn.readlane.i32(i32 %sgpr, i32 7)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: should optimize this as for i32
; CHECK-LABEL: {{^}}test_readlane_copy_from_sgpr_i64:
; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}
; CHECK: ;;#ASMEND
; CHECK: v_readlane_b32 [[SGPR0:s[0-9]+]], {{v[0-9]+}}, 7
; CHECK: v_readlane_b32 [[SGPR1:s[0-9]+]], {{v[0-9]+}}, 7
; CHECK: v_mov_b32_e32 {{v[0-9]+}}, [[SGPR0]]
; CHECK: v_mov_b32_e32 {{v[0-9]+}}, [[SGPR1]]
; CHECK: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @test_readlane_copy_from_sgpr_i64(ptr addrspace(1) %out) #1 {
  %sgpr = call i64 asm "s_mov_b64 $0, 0", "=s"()
  %readfirstlane = call i64 @llvm.amdgcn.readlane.i64(i64 %sgpr, i32 7)
  store i64 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: should optimize this as for i32
; CHECK-LABEL: {{^}}test_readlane_copy_from_sgpr_f64:
; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}
; CHECK: ;;#ASMEND
; CHECK: v_readlane_b32 [[SGPR0:s[0-9]+]], {{v[0-9]+}}, 7
; CHECK: v_readlane_b32 [[SGPR1:s[0-9]+]], {{v[0-9]+}}, 7
; CHECK: v_mov_b32_e32 {{v[0-9]+}}, [[SGPR0]]
; CHECK: v_mov_b32_e32 {{v[0-9]+}}, [[SGPR1]]
; CHECK: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @test_readlane_copy_from_sgpr_f64(ptr addrspace(1) %out) #1 {
  %sgpr = call double asm "s_mov_b64 $0, 0", "=s"()
  %readfirstlane = call double @llvm.amdgcn.readlane.f64(double %sgpr, i32 7)
  store double %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }
