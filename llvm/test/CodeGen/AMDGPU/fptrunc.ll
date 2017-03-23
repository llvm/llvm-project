; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN-UNSAFE %s

; FUNC-LABEL: {{^}}fptrunc_f64_to_f32:
; GCN: v_cvt_f32_f64_e32 {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fptrunc_f64_to_f32(float addrspace(1)* %out, double %in) {
  %result = fptrunc double %in to float
  store float %result, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fptrunc_f64_to_f16:
; GCN-NOT: v_cvt
; GCN-UNSAFE: v_cvt_f32_f64_e32 [[F32:v[0-9]+]]
; GCN-UNSAFE: v_cvt_f16_f32_e32 v{{[0-9]+}}, [[F32]]
define amdgpu_kernel void @fptrunc_f64_to_f16(i16 addrspace(1)* %out, double %in) {
  %result = fptrunc double %in to half
  %result_i16 = bitcast half %result to i16
  store i16 %result_i16, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fptrunc_v2f64_to_v2f32:
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
define amdgpu_kernel void @fptrunc_v2f64_to_v2f32(<2 x float> addrspace(1)* %out, <2 x double> %in) {
  %result = fptrunc <2 x double> %in to <2 x float>
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fptrunc_v4f64_to_v4f32:
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
define amdgpu_kernel void @fptrunc_v4f64_to_v4f32(<4 x float> addrspace(1)* %out, <4 x double> %in) {
  %result = fptrunc <4 x double> %in to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fptrunc_v8f64_to_v8f32:
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
; GCN: v_cvt_f32_f64_e32
define amdgpu_kernel void @fptrunc_v8f64_to_v8f32(<8 x float> addrspace(1)* %out, <8 x double> %in) {
  %result = fptrunc <8 x double> %in to <8 x float>
  store <8 x float> %result, <8 x float> addrspace(1)* %out
  ret void
}
