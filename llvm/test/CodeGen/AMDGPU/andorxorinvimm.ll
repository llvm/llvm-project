; RUN: llc -march=amdgcn -mcpu=tahiti -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}s_or_to_orn2:
; SI: s_orn2_b32 s{{[0-9]+}}, s{{[0-9]+}}, 50
define amdgpu_kernel void @s_or_to_orn2(i32 addrspace(1)* %out, i32 %in) {
  %x = or i32 %in, -51
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_or_to_orn2_imm0:
; SI: s_orn2_b32 s{{[0-9]+}}, s{{[0-9]+}}, 50
define amdgpu_kernel void @s_or_to_orn2_imm0(i32 addrspace(1)* %out, i32 %in) {
  %x = or i32 -51, %in
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_and_to_andn2:
; SI: s_andn2_b32 s{{[0-9]+}}, s{{[0-9]+}}, 50
define amdgpu_kernel void @s_and_to_andn2(i32 addrspace(1)* %out, i32 %in) {
  %x = and i32 %in, -51
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_and_to_andn2_imm0:
; SI: s_andn2_b32 s{{[0-9]+}}, s{{[0-9]+}}, 50
define amdgpu_kernel void @s_and_to_andn2_imm0(i32 addrspace(1)* %out, i32 %in) {
  %x = and i32 -51, %in
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_xor_to_xnor:
; SI: s_xnor_b32 s{{[0-9]+}}, s{{[0-9]+}}, 50
define amdgpu_kernel void @s_xor_to_xnor(i32 addrspace(1)* %out, i32 %in) {
  %x = xor i32 %in, -51
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_xor_to_xnor_imm0:
; SI: s_xnor_b32 s{{[0-9]+}}, s{{[0-9]+}}, 50
define amdgpu_kernel void @s_xor_to_xnor_imm0(i32 addrspace(1)* %out, i32 %in) {
  %x = xor i32 -51, %in
  store i32 %x, i32 addrspace(1)* %out
  ret void
}
