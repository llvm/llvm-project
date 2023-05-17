; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=SI %s

; FUNC-LABEL: {{^}}fmul_f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fmul_f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                      ptr addrspace(1) %in2) {
   %r0 = load double, ptr addrspace(1) %in1
   %r1 = load double, ptr addrspace(1) %in2
   %r2 = fmul double %r0, %r1
   store double %r2, ptr addrspace(1) %out
   ret void
}

; FUNC-LABEL: {{^}}fmul_v2f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fmul_v2f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                        ptr addrspace(1) %in2) {
   %r0 = load <2 x double>, ptr addrspace(1) %in1
   %r1 = load <2 x double>, ptr addrspace(1) %in2
   %r2 = fmul <2 x double> %r0, %r1
   store <2 x double> %r2, ptr addrspace(1) %out
   ret void
}

; FUNC-LABEL: {{^}}fmul_v4f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fmul_v4f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                        ptr addrspace(1) %in2) {
   %r0 = load <4 x double>, ptr addrspace(1) %in1
   %r1 = load <4 x double>, ptr addrspace(1) %in2
   %r2 = fmul <4 x double> %r0, %r1
   store <4 x double> %r2, ptr addrspace(1) %out
   ret void
}
