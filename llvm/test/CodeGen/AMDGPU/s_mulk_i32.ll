; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}s_mulk_i32_k0:
; SI: s_load_dword [[VAL:s[0-9]+]]
; SI: s_mulk_i32 [[VAL]], 0x41
; SI: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[VRESULT]]
; SI: s_endpgm
define void @s_mulk_i32_k0(i32 addrspace(1)* %out, i32 %b) {
  %mul = mul i32 %b, 65
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_mulk_i32_k1:
; SI: s_mulk_i32 {{s[0-9]+}}, 0x7fff{{$}}
; SI: s_endpgm
define void @s_mulk_i32_k1(i32 addrspace(1)* %out, i32 %b) {
  %mul = mul i32 %b, 32767 ; (1 << 15) - 1
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_mulk_i32_k2:
; SI: s_mulk_i32 {{s[0-9]+}}, 0xffef{{$}}
; SI: s_endpgm
define void @s_mulk_i32_k2(i32 addrspace(1)* %out, i32 %b) {
  %mul = mul i32 %b, -17
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}no_s_mulk_i32_k0:
; SI: s_mul_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x8001{{$}}
; SI: s_endpgm
define void @no_s_mulk_i32_k0(i32 addrspace(1)* %out, i32 %b) {
  %mul = mul i32 %b, 32769 ; 1 << 15 + 1
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}
