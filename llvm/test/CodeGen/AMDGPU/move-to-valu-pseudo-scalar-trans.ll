; RUN: llc -march=amdgcn -mcpu=gfx1200 -stop-after=si-fix-sgpr-copies -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: name: exp_f32
; CHECK: V_EXP_F32_e64
define amdgpu_kernel void @exp_f32(ptr addrspace(1) %ptr) {
  %val = load volatile float, ptr addrspace(1) %ptr
  %res = call float @llvm.exp.f32(float %val)
  store float %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: exp_f16
; CHECK: V_EXP_F16_e64
define amdgpu_kernel void @exp_f16(ptr addrspace(1) %ptr) {
  %val = load volatile half, ptr addrspace(1) %ptr
  %res = call half @llvm.exp.f16(half %val)
  store half %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: log_f32
; CHECK: V_LOG_F32_e64
define amdgpu_kernel void @log_f32(ptr addrspace(1) %ptr) {
  %val = load volatile float, ptr addrspace(1) %ptr
  %res = call float @llvm.log.f32(float %val)
  store float %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: log_f16
; CHECK: V_LOG_F16_e64
define amdgpu_kernel void @log_f16(ptr addrspace(1) %ptr) {
  %val = load volatile half, ptr addrspace(1) %ptr
  %res = call half @llvm.log.f16(half %val)
  store half %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: rcp_f32
; CHECK: V_RCP_F32_e64
define amdgpu_kernel void @rcp_f32(ptr addrspace(1) %ptr) {
  %val = load volatile float, ptr addrspace(1) %ptr
  %res = call float @llvm.amdgcn.rcp.f32(float %val)
  store float %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: rcp_f16
; CHECK: V_RCP_F16_e64
define amdgpu_kernel void @rcp_f16(ptr addrspace(1) %ptr) {
  %val = load volatile half, ptr addrspace(1) %ptr
  %res = call half @llvm.amdgcn.rcp.f16(half %val)
  store half %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: rsq_f32
; CHECK: V_RSQ_F32_e64
define amdgpu_kernel void @rsq_f32(ptr addrspace(1) %ptr) {
  %val = load volatile float, ptr addrspace(1) %ptr
  %res = call float @llvm.amdgcn.rsq.f32(float %val)
  store float %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: rsq_f16
; CHECK: V_RSQ_F16_e64
define amdgpu_kernel void @rsq_f16(ptr addrspace(1) %ptr) {
  %val = load volatile half, ptr addrspace(1) %ptr
  %res = call half @llvm.amdgcn.rsq.f16(half %val)
  store half %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: sqrt_f32
; CHECK: V_SQRT_F32_e64
define amdgpu_kernel void @sqrt_f32(ptr addrspace(1) %ptr) {
  %val = load volatile float, ptr addrspace(1) %ptr
  %res = call float @llvm.amdgcn.sqrt.f32(float %val)
  store float %res, ptr addrspace(1) %ptr
  ret void
}

; CHECK-LABEL: name: sqrt_f16
; CHECK: V_SQRT_F16_e64
define amdgpu_kernel void @sqrt_f16(ptr addrspace(1) %ptr) {
  %val = load volatile half, ptr addrspace(1) %ptr
  %res = call half @llvm.amdgcn.sqrt.f16(half %val)
  store half %res, ptr addrspace(1) %ptr
  ret void
}

declare float @llvm.exp.f32(float)
declare half @llvm.exp.f16(half)
declare float @llvm.log.f32(float)
declare half @llvm.log.f16(half)
declare float @llvm.amdgcn.rcp.f32(float)
declare half @llvm.amdgcn.rcp.f16(half)
declare float @llvm.amdgcn.rsq.f32(float)
declare half @llvm.amdgcn.rsq.f16(half)
declare float @llvm.amdgcn.sqrt.f32(float)
declare half @llvm.amdgcn.sqrt.f16(half)
