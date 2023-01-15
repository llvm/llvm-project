; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}kernel_ieee_mode_default:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[QUIET0:v[0-9]+]], 1.0, [[VAL0]]
; GCN-DAG: v_mul_f32_e32 [[QUIET1:v[0-9]+]], 1.0, [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[QUIET0]], [[QUIET1]]
; GCN-NOT: v_mul_f32
define amdgpu_kernel void @kernel_ieee_mode_default() #0 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}kernel_ieee_mode_on:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[QUIET0:v[0-9]+]], 1.0, [[VAL0]]
; GCN-DAG: v_mul_f32_e32 [[QUIET1:v[0-9]+]], 1.0, [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[QUIET0]], [[QUIET1]]
; GCN-NOT: v_mul_f32
define amdgpu_kernel void @kernel_ieee_mode_on() #1 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}kernel_ieee_mode_off:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-NOT: [[VAL0]]
; GCN-NOT: [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[VAL0]], [[VAL1]]
; GCN-NOT: v_mul_f32
define amdgpu_kernel void @kernel_ieee_mode_off() #2 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}func_ieee_mode_default:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[QUIET0:v[0-9]+]], 1.0, [[VAL0]]
; GCN-DAG: v_mul_f32_e32 [[QUIET1:v[0-9]+]], 1.0, [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[QUIET0]], [[QUIET1]]
; GCN-NOT: v_mul_f32
define void @func_ieee_mode_default() #0 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}func_ieee_mode_on:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[QUIET0:v[0-9]+]], 1.0, [[VAL0]]
; GCN-DAG: v_mul_f32_e32 [[QUIET1:v[0-9]+]], 1.0, [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[QUIET0]], [[QUIET1]]
; GCN-NOT: v_mul_f32
define void @func_ieee_mode_on() #1 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}func_ieee_mode_off:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-NOT: [[VAL0]]
; GCN-NOT: [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[VAL0]], [[VAL1]]
; GCN-NOT: v_mul_f32
define void @func_ieee_mode_off() #2 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}cs_ieee_mode_default:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-NOT: [[VAL0]]
; GCN-NOT: [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[QUIET0]], [[QUIET1]]
; GCN-NOT: v_mul_f32
define amdgpu_cs void @cs_ieee_mode_default() #0 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}cs_ieee_mode_on:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[QUIET0:v[0-9]+]], 1.0, [[VAL0]]
; GCN-DAG: v_mul_f32_e32 [[QUIET1:v[0-9]+]], 1.0, [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[QUIET0]], [[QUIET1]]
; GCN-NOT: v_mul_f32
define amdgpu_cs void @cs_ieee_mode_on() #1 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}cs_ieee_mode_off:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-NOT: [[VAL0]]
; GCN-NOT: [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[VAL0]], [[VAL1]]
; GCN-NOT: v_mul_f32
define amdgpu_cs void @cs_ieee_mode_off() #2 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}ps_ieee_mode_default:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-NOT: [[VAL0]]
; GCN-NOT: [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[VAL0]], [[VAL1]]
; GCN-NOT: v_mul_f32
define amdgpu_ps void @ps_ieee_mode_default() #0 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}ps_ieee_mode_on:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[QUIET0:v[0-9]+]], 1.0, [[VAL0]]
; GCN-DAG: v_mul_f32_e32 [[QUIET1:v[0-9]+]], 1.0, [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[QUIET0]], [[QUIET1]]
; GCN-NOT: v_mul_f32
define amdgpu_ps void @ps_ieee_mode_on() #1 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}ps_ieee_mode_off:
; GCN: {{buffer|global|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|global|flat}}_load_dword [[VAL1:v[0-9]+]]
; GCN-NOT: [[VAL0]]
; GCN-NOT: [[VAL1]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], [[VAL0]], [[VAL1]]
; GCN-NOT: v_mul_f32
define amdgpu_ps void @ps_ieee_mode_off() #2 {
  %val0 = load volatile float, ptr addrspace(1) undef
  %val1 = load volatile float, ptr addrspace(1) undef
  %min = call float @llvm.minnum.f32(float %val0, float %val1)
  store volatile float %min, ptr addrspace(1) undef
  ret void
}

declare float @llvm.minnum.f32(float, float) #3

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-ieee"="true" }
attributes #2 = { nounwind "amdgpu-ieee"="false" }
attributes #3 = { nounwind readnone speculatable }
