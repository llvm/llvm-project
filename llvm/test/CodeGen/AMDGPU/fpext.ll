; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}fpext_f32_to_f64:
; SI: v_cvt_f64_f32_e32 {{v\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
define void @fpext_f32_to_f64(double addrspace(1)* %out, float %in) {
  %result = fpext float %in to double
  store double %result, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fpext_v2f32_to_v2f64:
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
define void @fpext_v2f32_to_v2f64(<2 x double> addrspace(1)* %out, <2 x float> %in) {
  %result = fpext <2 x float> %in to <2 x double>
  store <2 x double> %result, <2 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fpext_v3f32_to_v3f64:
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
define void @fpext_v3f32_to_v3f64(<3 x double> addrspace(1)* %out, <3 x float> %in) {
  %result = fpext <3 x float> %in to <3 x double>
  store <3 x double> %result, <3 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fpext_v4f32_to_v4f64:
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
define void @fpext_v4f32_to_v4f64(<4 x double> addrspace(1)* %out, <4 x float> %in) {
  %result = fpext <4 x float> %in to <4 x double>
  store <4 x double> %result, <4 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fpext_v8f32_to_v8f64:
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
define void @fpext_v8f32_to_v8f64(<8 x double> addrspace(1)* %out, <8 x float> %in) {
  %result = fpext <8 x float> %in to <8 x double>
  store <8 x double> %result, <8 x double> addrspace(1)* %out
  ret void
}
