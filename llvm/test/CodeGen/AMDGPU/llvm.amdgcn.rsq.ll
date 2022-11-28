; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.amdgcn.rsq.f32(float) #0
declare double @llvm.amdgcn.rsq.f64(double) #0

; FUNC-LABEL: {{^}}rsq_f32:
; SI: v_rsq_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @rsq_f32(ptr addrspace(1) %out, float %src) #1 {
  %rsq = call float @llvm.amdgcn.rsq.f32(float %src) #0
  store float %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: Really these should be constant folded
; FUNC-LABEL: {{^}}rsq_f32_constant_4.0
; SI: v_rsq_f32_e32 {{v[0-9]+}}, 4.0
define amdgpu_kernel void @rsq_f32_constant_4.0(ptr addrspace(1) %out) #1 {
  %rsq = call float @llvm.amdgcn.rsq.f32(float 4.0) #0
  store float %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_f32_constant_100.0
; SI: v_rsq_f32_e32 {{v[0-9]+}}, 0x42c80000
define amdgpu_kernel void @rsq_f32_constant_100.0(ptr addrspace(1) %out) #1 {
  %rsq = call float @llvm.amdgcn.rsq.f32(float 100.0) #0
  store float %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_f64:
; SI: v_rsq_f64_e32 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @rsq_f64(ptr addrspace(1) %out, double %src) #1 {
  %rsq = call double @llvm.amdgcn.rsq.f64(double %src) #0
  store double %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: Really these should be constant folded
; FUNC-LABEL: {{^}}rsq_f64_constant_4.0
; SI: v_rsq_f64_e32 {{v\[[0-9]+:[0-9]+\]}}, 4.0
define amdgpu_kernel void @rsq_f64_constant_4.0(ptr addrspace(1) %out) #1 {
  %rsq = call double @llvm.amdgcn.rsq.f64(double 4.0) #0
  store double %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_f64_constant_100.0
; SI-DAG: s_mov_b32 s{{[0-9]+}}, 0x40590000
; SI-DAG: s_mov_b32 s{{[0-9]+}}, 0{{$}}
; SI: v_rsq_f64_e32 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @rsq_f64_constant_100.0(ptr addrspace(1) %out) #1 {
  %rsq = call double @llvm.amdgcn.rsq.f64(double 100.0) #0
  store double %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_undef_f32:
; SI-NOT: v_rsq_f32
define amdgpu_kernel void @rsq_undef_f32(ptr addrspace(1) %out) #1 {
  %rsq = call float @llvm.amdgcn.rsq.f32(float undef)
  store float %rsq, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
