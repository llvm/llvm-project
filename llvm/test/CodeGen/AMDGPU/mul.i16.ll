; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,GFX89 %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX89 %s

; GCN-LABEL: {{^}}v_mul_i16:
; SI: s_mov_b32 [[K:s[0-9]+]], 0xffff{{$}}
; SI: v_and_b32_e32 v{{[0-9]+}}, [[K]]
; SI: v_and_b32_e32 v{{[0-9]+}}, [[K]]
; SI: v_mul_u32_u24

; GFX89: v_mul_lo_u16_e32 v0, v0, v1
define i16 @v_mul_i16(i16 %a, i16 %b) {
  %r.val = mul i16 %a, %b
  ret i16 %r.val
}

; FIXME: Should emit scalar mul or maybe i16 v_mul here
; GCN-LABEL: {{^}}s_mul_i16:
; SI: v_mul_u32_u24
; VI: s_mul_i16
define amdgpu_kernel void @s_mul_i16(i16 %a, i16 %b) {
  %r.val = mul i16 %a, %b
  store volatile i16 %r.val, i16 addrspace(1)* null
  ret void
}

; FIXME: Should emit u16 mul here. Instead it's worse than SI
; GCN-LABEL: {{^}}v_mul_i16_uniform_load:
; SI: v_mul_u32_u24
; GFX89: v_mul_lo_u32
define amdgpu_kernel void @v_mul_i16_uniform_load(
    i16 addrspace(1)* %r,
    i16 addrspace(1)* %a,
    i16 addrspace(1)* %b) {
entry:
  %a.val = load i16, i16 addrspace(1)* %a
  %b.val = load i16, i16 addrspace(1)* %b
  %r.val = mul i16 %a.val, %b.val
  store i16 %r.val, i16 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}v_mul_v2i16:
; SI: v_mul_u32_u24
; SI: v_mul_u32_u24

; VI: v_mul_lo_u16_sdwa
; VI: v_mul_lo_u16_e32
; VI: v_or_b32_e32


; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_mul_lo_u16 v0, v0, v1
; GFX9-NEXT: s_setpc_b64
define <2 x i16> @v_mul_v2i16(<2 x i16> %a, <2 x i16> %b) {
  %r.val = mul <2 x i16> %a, %b
  ret <2 x i16> %r.val
}

; FIXME: Unpack garbage on gfx9
; GCN-LABEL: {{^}}v_mul_v3i16:
; SI: v_mul_u32_u24
; SI: v_mul_u32_u24
; SI: v_mul_u32_u24

; VI: v_mul_lo_u16
; VI: v_mul_lo_u16
; VI: v_mul_lo_u16

; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_mul_lo_u16
; GFX9-NEXT: v_pk_mul_lo_u16
; GFX9-NEXT: s_setpc_b64
define <3 x i16> @v_mul_v3i16(<3 x i16> %a, <3 x i16> %b) {
  %r.val = mul <3 x i16> %a, %b
  ret <3 x i16> %r.val
}

; GCN-LABEL: {{^}}v_mul_v4i16:
; SI: v_mul_u32_u24
; SI: v_mul_u32_u24
; SI: v_mul_u32_u24
; SI: v_mul_u32_u24

; VI: v_mul_lo_u16_sdwa
; VI: v_mul_lo_u16_e32
; VI: v_mul_lo_u16_sdwa
; VI: v_mul_lo_u16_e32
; VI: v_or_b32_e32
; VI: v_or_b32_e32

; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_mul_lo_u16 v0, v0, v2
; GFX9-NEXT: v_pk_mul_lo_u16 v1, v1, v3
; GFX9-NEXT: s_setpc_b64
define <4 x i16> @v_mul_v4i16(<4 x i16> %a, <4 x i16> %b) {
  %r.val = mul <4 x i16> %a, %b
  ret <4 x i16> %r.val
}
