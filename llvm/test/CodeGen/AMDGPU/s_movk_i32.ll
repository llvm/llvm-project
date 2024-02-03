; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}s_movk_i32_k0:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0xffff, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 1, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k0(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 4295032831 ; ((1 << 16) - 1) | (1 << 32)
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 4295032831)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k1:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x7fff, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 1, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k1(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 4295000063 ; ((1 << 15) - 1) | (1 << 32)
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 4295000063)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k2:
; SI-DAG: s_movk_i32 [[LO_S_IMM:s[0-9]+]], 0x7fff{{$}}
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x7fff, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 64, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k2(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 274877939711 ; ((1 << 15) - 1) | (64 << 32)
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 274877939711)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k3:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x8000, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 1, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k3(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 4295000064 ; (1 << 15) | (1 << 32)
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 4295000064)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k4:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x20000, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 1, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k4(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 4295098368 ; (1 << 17) | (1 << 32)
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 4295098368)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k5:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0xffffffef, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0xff00ffff, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k5(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 18374967954648334319 ; -17 & 0xff00ffffffffffff
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 18374967954648334319)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k6:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x41, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 63, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k6(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 270582939713 ; 65 | (63 << 32)
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 270582939713)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k7:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x2000, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x4000, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k7(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 70368744185856; ((1 << 13)) | ((1 << 14) << 32)
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 70368744185856)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k8:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0xffff8000, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x11111111, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k8(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 1229782942255906816 ; 0x11111111ffff8000
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 1229782942255906816)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k9:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0xffff8001, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x11111111, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k9(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 1229782942255906817 ; 0x11111111ffff8001
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 1229782942255906817)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k10:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0xffff8888, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x11111111, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k10(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 1229782942255909000 ; 0x11111111ffff8888
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 1229782942255909000)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k11:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0xffff8fff, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x11111111, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k11(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 1229782942255910911 ; 0x11111111ffff8fff
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 1229782942255910911)
  ret void
}

; SI-LABEL: {{^}}s_movk_i32_k12:
; SI-DAG: buffer_load_dwordx2 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]],
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0xffff7001, v[[LO_VREG]]
; SI-DAG: v_or_b32_e32 {{v[0-9]+}}, 0x11111111, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @s_movk_i32_k12(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 4
  %or = or i64 %loada, 1229782942255902721 ; 0x11111111ffff7001
  store i64 %or, ptr addrspace(1) %out
  call void asm sideeffect "; use $0", "s"(i64 1229782942255902721)
  ret void
}
