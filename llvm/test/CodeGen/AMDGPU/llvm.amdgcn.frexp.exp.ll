; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s  | FileCheck -check-prefix=GCN %s

declare float @llvm.fabs.f32(float) #0
declare double @llvm.fabs.f64(double) #0
declare i32 @llvm.amdgcn.frexp.exp.f32(float) #0
declare i32 @llvm.amdgcn.frexp.exp.f64(double) #0

; GCN-LABEL: {{^}}s_test_frexp_exp_f32:
; GCN: v_frexp_exp_i32_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define void @s_test_frexp_exp_f32(i32 addrspace(1)* %out, float %src) #1 {
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.f32(float %src)
  store i32 %frexp.exp, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fabs_frexp_exp_f32:
; GCN: v_frexp_exp_i32_f32_e64 {{v[0-9]+}}, |{{s[0-9]+}}|
define void @s_test_fabs_frexp_exp_f32(i32 addrspace(1)* %out, float %src) #1 {
  %fabs.src = call float @llvm.fabs.f32(float %src)
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.f32(float %fabs.src)
  store i32 %frexp.exp, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fneg_fabs_frexp_exp_f32:
; GCN: v_frexp_exp_i32_f32_e64 {{v[0-9]+}}, -|{{s[0-9]+}}|
define void @s_test_fneg_fabs_frexp_exp_f32(i32 addrspace(1)* %out, float %src) #1 {
  %fabs.src = call float @llvm.fabs.f32(float %src)
  %fneg.fabs.src = fsub float -0.0, %fabs.src
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.f32(float %fneg.fabs.src)
  store i32 %frexp.exp, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_frexp_exp_f64:
; GCN: v_frexp_exp_i32_f64_e32 {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}
define void @s_test_frexp_exp_f64(i32 addrspace(1)* %out, double %src) #1 {
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.f64(double %src)
  store i32 %frexp.exp, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fabs_frexp_exp_f64:
; GCN: v_frexp_exp_i32_f64_e64 {{v[0-9]+}}, |{{s\[[0-9]+:[0-9]+\]}}|
define void @s_test_fabs_frexp_exp_f64(i32 addrspace(1)* %out, double %src) #1 {
  %fabs.src = call double @llvm.fabs.f64(double %src)
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.f64(double %fabs.src)
  store i32 %frexp.exp, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_fneg_fabs_frexp_exp_f64:
; GCN: v_frexp_exp_i32_f64_e64 {{v[0-9]+}}, -|{{s\[[0-9]+:[0-9]+\]}}|
define void @s_test_fneg_fabs_frexp_exp_f64(i32 addrspace(1)* %out, double %src) #1 {
  %fabs.src = call double @llvm.fabs.f64(double %src)
  %fneg.fabs.src = fsub double -0.0, %fabs.src
  %frexp.exp = call i32 @llvm.amdgcn.frexp.exp.f64(double %fneg.fabs.src)
  store i32 %frexp.exp, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
