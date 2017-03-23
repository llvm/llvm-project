; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}v_fsub_f32:
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define amdgpu_kernel void @v_fsub_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %a = load float, float addrspace(1)* %in, align 4
  %b = load float, float addrspace(1)* %b_ptr, align 4
  %result = fsub float %a, %b
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_fsub_f32:
; R600: ADD {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, -KC0[2].W

; SI: v_sub_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
define amdgpu_kernel void @s_fsub_f32(float addrspace(1)* %out, float %a, float %b) {
  %sub = fsub float %a, %b
  store float %sub, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}fsub_v2f32:
; R600-DAG: ADD {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, -KC0[3].Z
; R600-DAG: ADD {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, -KC0[3].Y

; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
define amdgpu_kernel void @fsub_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) {
  %sub = fsub <2 x float> %a, %b
  store <2 x float> %sub, <2 x float> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_fsub_v4f32:
; R600: ADD {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}
; R600: ADD {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}
; R600: ADD {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}
; R600: ADD {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}

; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define amdgpu_kernel void @v_fsub_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float>, <4 x float> addrspace(1)* %in, align 16
  %b = load <4 x float>, <4 x float> addrspace(1)* %b_ptr, align 16
  %result = fsub <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}s_fsub_v4f32:
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; SI: v_subrev_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; SI: s_endpgm
define amdgpu_kernel void @s_fsub_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %a, <4 x float> %b) {
  %result = fsub <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_fsub_f32:
; SI: v_subrev_f32_e32 [[SUB:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; SI: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[SUB]]
define amdgpu_kernel void @v_fneg_fsub_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %a = load float, float addrspace(1)* %in, align 4
  %b = load float, float addrspace(1)* %b_ptr, align 4
  %result = fsub float %a, %b
  %neg.result = fsub float -0.0, %result
  store float %neg.result, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_fsub_nsz_f32:
; SI: v_subrev_f32_e32 [[SUB:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; SI-NOT: xor
define amdgpu_kernel void @v_fneg_fsub_nsz_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %a = load float, float addrspace(1)* %in, align 4
  %b = load float, float addrspace(1)* %b_ptr, align 4
  %result = fsub nsz float %a, %b
  %neg.result = fsub float -0.0, %result
  store float %neg.result, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_fsub_nsz_attribute_f32:
; SI: v_subrev_f32_e32 [[SUB:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; SI-NOT: xor
define amdgpu_kernel void @v_fneg_fsub_nsz_attribute_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %a = load float, float addrspace(1)* %in, align 4
  %b = load float, float addrspace(1)* %b_ptr, align 4
  %result = fsub float %a, %b
  %neg.result = fsub float -0.0, %result
  store float %neg.result, float addrspace(1)* %out, align 4
  ret void
}

; For some reason the attribute has a string "true" or "false", so
; make sure it is disabled and the fneg is not folded if it is not
; "true".
; FUNC-LABEL: {{^}}v_fneg_fsub_nsz_false_attribute_f32:
; SI: v_subrev_f32_e32 [[SUB:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; SI: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[SUB]]
define amdgpu_kernel void @v_fneg_fsub_nsz_false_attribute_f32(float addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %a = load float, float addrspace(1)* %in, align 4
  %b = load float, float addrspace(1)* %b_ptr, align 4
  %result = fsub float %a, %b
  %neg.result = fsub float -0.0, %result
  store float %neg.result, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_fsub_0_nsz_attribute_f32:
; SI-NOT: v_sub
define amdgpu_kernel void @v_fsub_0_nsz_attribute_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %a = load float, float addrspace(1)* %in, align 4
  %result = fsub float %a, 0.0
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind "no-signed-zeros-fp-math"="true" }
attributes #1 = { nounwind "no-signed-zeros-fp-math"="false" }
