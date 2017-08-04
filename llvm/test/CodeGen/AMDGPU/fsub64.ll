; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare double @llvm.fabs.f64(double) #0

; SI-LABEL: {{^}}fsub_f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fsub_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                      double addrspace(1)* %in2) {
  %r0 = load double, double addrspace(1)* %in1
  %r1 = load double, double addrspace(1)* %in2
  %r2 = fsub double %r0, %r1
  store double %r2, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}fsub_fabs_f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|}}
define amdgpu_kernel void @fsub_fabs_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                           double addrspace(1)* %in2) {
  %r0 = load double, double addrspace(1)* %in1
  %r1 = load double, double addrspace(1)* %in2
  %r1.fabs = call double @llvm.fabs.f64(double %r1) #0
  %r2 = fsub double %r0, %r1.fabs
  store double %r2, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}fsub_fabs_inv_f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], |v\[[0-9]+:[0-9]+\]|, -v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fsub_fabs_inv_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                               double addrspace(1)* %in2) {
  %r0 = load double, double addrspace(1)* %in1
  %r1 = load double, double addrspace(1)* %in2
  %r0.fabs = call double @llvm.fabs.f64(double %r0) #0
  %r2 = fsub double %r0.fabs, %r1
  store double %r2, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_fsub_f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @s_fsub_f64(double addrspace(1)* %out, double %a, double %b) {
  %sub = fsub double %a, %b
  store double %sub, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_fsub_imm_f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], -s\[[0-9]+:[0-9]+\]}}, 4.0
define amdgpu_kernel void @s_fsub_imm_f64(double addrspace(1)* %out, double %a, double %b) {
  %sub = fsub double 4.0, %a
  store double %sub, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_fsub_imm_inv_f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\]}}, -4.0
define amdgpu_kernel void @s_fsub_imm_inv_f64(double addrspace(1)* %out, double %a, double %b) {
  %sub = fsub double %a, 4.0
  store double %sub, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_fsub_self_f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], -s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @s_fsub_self_f64(double addrspace(1)* %out, double %a) {
  %sub = fsub double %a, %a
  store double %sub, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}fsub_v2f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fsub_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %a, <2 x double> %b) {
  %sub = fsub <2 x double> %a, %b
  store <2 x double> %sub, <2 x double> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}fsub_v4f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fsub_v4f64(<4 x double> addrspace(1)* %out, <4 x double> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x double>, <4 x double> addrspace(1)* %in, i32 1
  %a = load <4 x double>, <4 x double> addrspace(1)* %in
  %b = load <4 x double>, <4 x double> addrspace(1)* %b_ptr
  %result = fsub <4 x double> %a, %b
  store <4 x double> %result, <4 x double> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_fsub_v4f64:
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
; SI: v_add_f64 {{v\[[0-9]+:[0-9]+\], s\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @s_fsub_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %a, <4 x double> %b) {
  %result = fsub <4 x double> %a, %b
  store <4 x double> %result, <4 x double> addrspace(1)* %out, align 16
  ret void
}

attributes #0 = { nounwind readnone }
