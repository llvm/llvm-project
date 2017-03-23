; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=SI %s

; FUNC-LABEL: {{^}}fmul_f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fmul_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                      double addrspace(1)* %in2) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = fmul double %r0, %r1
   store double %r2, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fmul_v2f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fmul_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %in1,
                        <2 x double> addrspace(1)* %in2) {
   %r0 = load <2 x double>, <2 x double> addrspace(1)* %in1
   %r1 = load <2 x double>, <2 x double> addrspace(1)* %in2
   %r2 = fmul <2 x double> %r0, %r1
   store <2 x double> %r2, <2 x double> addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fmul_v4f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fmul_v4f64(<4 x double> addrspace(1)* %out, <4 x double> addrspace(1)* %in1,
                        <4 x double> addrspace(1)* %in2) {
   %r0 = load <4 x double>, <4 x double> addrspace(1)* %in1
   %r1 = load <4 x double>, <4 x double> addrspace(1)* %in2
   %r2 = fmul <4 x double> %r0, %r1
   store <4 x double> %r2, <4 x double> addrspace(1)* %out
   ret void
}
