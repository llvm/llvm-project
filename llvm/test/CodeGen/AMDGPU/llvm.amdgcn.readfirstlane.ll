; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope %s

declare i32 @llvm.amdgcn.readfirstlane(i32) #0
declare i64 @llvm.amdgcn.readfirstlane.i64(i64) #0
declare double @llvm.amdgcn.readfirstlane.f64(double) #0

; CHECK-LABEL: {{^}}test_readfirstlane_i32:
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, v2
define void @test_readfirstlane_i32(ptr addrspace(1) %out, i32 %src) #1 {
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %src)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_i64:
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
define void @test_readfirstlane_i64(ptr addrspace(1) %out, i64 %src) #1 {
  %readfirstlane = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %src)
  store i64 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_f64:
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; CHECK: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
define void @test_readfirstlane_f64(ptr addrspace(1) %out, double %src) #1 {
  %readfirstlane = call double @llvm.amdgcn.readfirstlane.f64(double %src)
  store double %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_imm_i32:
; CHECK: s_mov_b32 [[SGPR_VAL:s[0-9]]], 32
; CHECK-NOT: [[SGPR_VAL]]
; CHECK: ; use [[SGPR_VAL]]
define amdgpu_kernel void @test_readfirstlane_imm_i32(ptr addrspace(1) %out) #1 {
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane.i32(i32 32)
  call void asm sideeffect "; use $0", "s"(i32 %readfirstlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_imm_i64:
; CHECK: s_mov_b64 [[SGPR_VAL:s\[[0-9]+:[0-9]+\]]], 32
; CHECK: use [[SGPR_VAL]]
define amdgpu_kernel void @test_readfirstlane_imm_i64(ptr addrspace(1) %out) #1 {
  %readfirstlane = call i64 @llvm.amdgcn.readfirstlane.i64(i64 32)
  call void asm sideeffect "; use $0", "s"(i64 %readfirstlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_imm_f64:
; CHECK: s_mov_b32 s[[VAL0:[0-9]+]], 0
; CHECK: s_mov_b32 s[[VAL1:[0-9]+]], 0x40400000
; use s[[VAL0\:VAL1]]
define amdgpu_kernel void @test_readfirstlane_imm_f64(ptr addrspace(1) %out) #1 {
  %readfirstlane = call double @llvm.amdgcn.readfirstlane.f64(double 32.0)
  call void asm sideeffect "; use $0", "s"(double %readfirstlane)
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_imm_fold_i32:
; CHECK: v_mov_b32_e32 [[VVAL:v[0-9]]], 32
; CHECK-NOT: [[VVAL]]
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VVAL]]
define amdgpu_kernel void @test_readfirstlane_imm_fold_i32(ptr addrspace(1) %out) #1 {
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane.i32(i32 32)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_readfirstlane_imm_fold_i64:
; CHECK: s_mov_b64 s[[[VAL0:[0-9]+]]:[[VAL1:[0-9]+]]], 32
; CHECK: v_mov_b32_e32 v[[RES0:[0-9]+]], s[[VAL0]]
; CHECK: v_mov_b32_e32 v[[RES1:[0-9]+]], s[[VAL1]]
; CHECK: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[RES0]]:[[RES1]]]
define amdgpu_kernel void @test_readfirstlane_imm_fold_i64(ptr addrspace(1) %out) #1 {
  %readfirstlane = call i64 @llvm.amdgcn.readfirstlane.i64(i64 32)
  store i64 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: s_mov_b32 s[[VAL0:[0-9]+]], 0
; CHECK: s_mov_b32 s[[VAL1:[0-9]+]], 0x40400000
; CHECK: v_mov_b32_e32 v[[RES0:[0-9]+]], s[[VAL0]]
; CHECK: v_mov_b32_e32 v[[RES1:[0-9]+]], s[[VAL1]]
; CHECK: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[RES0]]:[[RES1]]]
define amdgpu_kernel void @test_readfirstlane_imm_fold_f64(ptr addrspace(1) %out) #1 {
  %readfirstlane = call double @llvm.amdgcn.readfirstlane.f64(double 32.0)
  store double %readfirstlane, ptr addrspace(1) %out, align 4
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

; CHECK-LABEL: {{^}}test_readfirstlane_copy_from_sgpr_i32:
; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_mov_b32 [[SGPR:s[0-9]+]]
; CHECK: ;;#ASMEND
; CHECK-NOT: [[SGPR]]
; CHECK-NOT: readfirstlane
; CHECK: v_mov_b32_e32 [[VCOPY:v[0-9]+]], [[SGPR]]
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[VCOPY]]
define amdgpu_kernel void @test_readfirstlane_copy_from_sgpr_i32(ptr addrspace(1) %out) #1 {
  %sgpr = call i32 asm "s_mov_b32 $0, 0", "=s"()
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %sgpr)
  store i32 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: should optimize this as for i32
; CHECK-LABEL: {{^}}test_readfirstlane_copy_from_sgpr_i64:
; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}
; CHECK: ;;#ASMEND
; CHECK: v_readfirstlane_b32 [[SGPR0:s[0-9]+]], {{v[0-9]+}}
; CHECK: v_readfirstlane_b32 [[SGPR1:s[0-9]+]], {{v[0-9]+}}
; CHECK: v_mov_b32_e32 {{v[0-9]+}}, [[SGPR0]]
; CHECK: v_mov_b32_e32 {{v[0-9]+}}, [[SGPR1]]
; CHECK: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @test_readfirstlane_copy_from_sgpr_i64(ptr addrspace(1) %out) #1 {
  %sgpr = call i64 asm "s_mov_b64 $0, 0", "=s"()
  %readfirstlane = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %sgpr)
  store i64 %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: should optimize this as for i32
; CHECK-LABEL: {{^}}test_readfirstlane_copy_from_sgpr_f64:
; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}
; CHECK: ;;#ASMEND
; CHECK: v_readfirstlane_b32 [[SGPR0:s[0-9]+]], {{v[0-9]+}}
; CHECK: v_readfirstlane_b32 [[SGPR1:s[0-9]+]], {{v[0-9]+}}
; CHECK: v_mov_b32_e32 {{v[0-9]+}}, [[SGPR0]]
; CHECK: v_mov_b32_e32 {{v[0-9]+}}, [[SGPR1]]
; CHECK: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @test_readfirstlane_copy_from_sgpr_f64(ptr addrspace(1) %out) #1 {
  %sgpr = call double asm "s_mov_b64 $0, 0", "=s"()
  %readfirstlane = call double @llvm.amdgcn.readfirstlane.f64(double %sgpr)
  store double %readfirstlane, ptr addrspace(1) %out, align 4
  ret void
}

; Make sure this doesn't crash.
; CHECK-LABEL: {{^}}test_readfirstlane_fi:
; CHECK: s_mov_b32 [[FIVAL:s[0-9]]], 0
define amdgpu_kernel void @test_readfirstlane_fi(ptr addrspace(1) %out) #1 {
  %alloca = alloca i32, addrspace(5)
  %int = ptrtoint ptr addrspace(5) %alloca to i32
  %readfirstlane = call i32 @llvm.amdgcn.readfirstlane(i32 %int)
  call void asm sideeffect "; use $0", "s"(i32 %readfirstlane)
  ret void
}

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
