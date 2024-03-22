; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s  | FileCheck -check-prefix=GCN %s

declare float @llvm.fabs.f32(float) nounwind readnone
declare float @llvm.copysign.f32(float, float) nounwind readnone
declare double @llvm.fabs.f64(double) nounwind readnone
declare i32 @llvm.amdgcn.frexp.exp.i32.f32(float) nounwind readnone
declare i32 @llvm.amdgcn.frexp.exp.i32.f64(double) nounwind readnone

; GCN-LABEL: {{^}}s_test_frexp_exp_f32:
; GCN: v_frexp_exp_i32_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @s_test_frexp_exp_f32(ptr addrspace(1) %out, float %src) nounwind {
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.i32.f32(float %src)
  store i32 %frexp.exp, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fabs_frexp_exp_f32:
; GCN: v_frexp_exp_i32_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @s_test_fabs_frexp_exp_f32(ptr addrspace(1) %out, float %src) nounwind {
  %fabs.src = call float @llvm.fabs.f32(float %src)
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.i32.f32(float %fabs.src)
  store i32 %frexp.exp, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fneg_fabs_frexp_exp_f32:
; GCN: v_frexp_exp_i32_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @s_test_fneg_fabs_frexp_exp_f32(ptr addrspace(1) %out, float %src) nounwind {
  %fabs.src = call float @llvm.fabs.f32(float %src)
  %fneg.fabs.src = fneg float %fabs.src
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.i32.f32(float %fneg.fabs.src)
  store i32 %frexp.exp, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}s_test_copysign_frexp_exp_f32:
; GCN: v_frexp_exp_i32_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @s_test_copysign_frexp_exp_f32(ptr addrspace(1) %out, float %src, float %sign) nounwind {
  %copysign = call float @llvm.copysign.f32(float %src, float %sign)
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.i32.f32(float %copysign)
  store i32 %frexp.exp, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}s_test_frexp_exp_f64:
; GCN: v_frexp_exp_i32_f64_e32 {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @s_test_frexp_exp_f64(ptr addrspace(1) %out, double %src) nounwind {
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.i32.f64(double %src)
  store i32 %frexp.exp, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fabs_frexp_exp_f64:
; GCN: v_frexp_exp_i32_f64_e32 {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @s_test_fabs_frexp_exp_f64(ptr addrspace(1) %out, double %src) nounwind {
  %fabs.src = call double @llvm.fabs.f64(double %src)
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.i32.f64(double %fabs.src)
  store i32 %frexp.exp, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fneg_fabs_frexp_exp_f64:
; GCN: v_frexp_exp_i32_f64_e32 {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @s_test_fneg_fabs_frexp_exp_f64(ptr addrspace(1) %out, double %src) nounwind {
  %fabs.src = call double @llvm.fabs.f64(double %src)
  %fneg.fabs.src = fneg double %fabs.src
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.i32.f64(double %fneg.fabs.src)
  store i32 %frexp.exp, ptr addrspace(1) %out
  ret void
}
