; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s  | FileCheck -check-prefix=GCN %s

declare float @llvm.fabs.f32(float) #0
declare double @llvm.fabs.f64(double) #0
declare float @llvm.amdgcn.frexp.mant.f32(float) #0
declare double @llvm.amdgcn.frexp.mant.f64(double) #0

; GCN-LABEL: {{^}}s_test_frexp_mant_f32:
; GCN: v_frexp_mant_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define void @s_test_frexp_mant_f32(float addrspace(1)* %out, float %src) #1 {
  %frexp.mant = call float @llvm.amdgcn.frexp.mant.f32(float %src)
  store float %frexp.mant, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fabs_frexp_mant_f32:
; GCN: v_frexp_mant_f32_e64 {{v[0-9]+}}, |{{s[0-9]+}}|
define void @s_test_fabs_frexp_mant_f32(float addrspace(1)* %out, float %src) #1 {
  %fabs.src = call float @llvm.fabs.f32(float %src)
  %frexp.mant = call float @llvm.amdgcn.frexp.mant.f32(float %fabs.src)
  store float %frexp.mant, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fneg_fabs_frexp_mant_f32:
; GCN: v_frexp_mant_f32_e64 {{v[0-9]+}}, -|{{s[0-9]+}}|
define void @s_test_fneg_fabs_frexp_mant_f32(float addrspace(1)* %out, float %src) #1 {
  %fabs.src = call float @llvm.fabs.f32(float %src)
  %fneg.fabs.src = fsub float -0.0, %fabs.src
  %frexp.mant = call float @llvm.amdgcn.frexp.mant.f32(float %fneg.fabs.src)
  store float %frexp.mant, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_frexp_mant_f64:
; GCN: v_frexp_mant_f64_e32 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @s_test_frexp_mant_f64(double addrspace(1)* %out, double %src) #1 {
  %frexp.mant = call double @llvm.amdgcn.frexp.mant.f64(double %src)
  store double %frexp.mant, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fabs_frexp_mant_f64:
; GCN: v_frexp_mant_f64_e64 {{v\[[0-9]+:[0-9]+\]}}, |{{s\[[0-9]+:[0-9]+\]}}|
define void @s_test_fabs_frexp_mant_f64(double addrspace(1)* %out, double %src) #1 {
  %fabs.src = call double @llvm.fabs.f64(double %src)
  %frexp.mant = call double @llvm.amdgcn.frexp.mant.f64(double %fabs.src)
  store double %frexp.mant, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fneg_fabs_frexp_mant_f64:
; GCN: v_frexp_mant_f64_e64 {{v\[[0-9]+:[0-9]+\]}}, -|{{s\[[0-9]+:[0-9]+\]}}|
define void @s_test_fneg_fabs_frexp_mant_f64(double addrspace(1)* %out, double %src) #1 {
  %fabs.src = call double @llvm.fabs.f64(double %src)
  %fneg.fabs.src = fsub double -0.0, %fabs.src
  %frexp.mant = call double @llvm.amdgcn.frexp.mant.f64(double %fneg.fabs.src)
  store double %frexp.mant, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
