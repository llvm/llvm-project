; RUN: llc -mtriple=amdgcn -mcpu=bonaire < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn -mcpu=carrizo -mattr=-flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=ALL %s

; ALL-LABEL: {{^}}large_alloca_pixel_shader:
; GCN-DAG: s_mov_b32 s4, SCRATCH_RSRC_DWORD0
; GCN-DAG: s_mov_b32 s5, SCRATCH_RSRC_DWORD1
; GCN-DAG: s_mov_b32 s6, -1

; CI-DAG: s_mov_b32 s7, 0xe8f000
; VI-DAG: s_mov_b32 s7, 0xe80000
; GFX9-DAG: s_mov_b32 s7, 0xe00000

; GCN: s_add_u32 s4, s4, s0
; GCN: s_addc_u32 s5, s5, 0

; GCN: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, s[4:7], 0 offen
; GCN: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, s[4:7], 0 offen

; ALL: ; ScratchSize: 32772
define amdgpu_ps void @large_alloca_pixel_shader(i32 %x, i32 %y) #0 {
  %large = alloca [8192 x i32], align 4, addrspace(5)
  %gep = getelementptr [8192 x i32], ptr addrspace(5) %large, i32 0, i32 8191
  store volatile i32 %x, ptr addrspace(5) %gep
  %gep1 = getelementptr [8192 x i32], ptr addrspace(5) %large, i32 0, i32 %y
  %val = load volatile i32, ptr addrspace(5) %gep1
  store volatile i32 %val, ptr addrspace(1) undef
  ret void
}

; ALL-LABEL: {{^}}large_alloca_pixel_shader_inreg:
; GCN-DAG: s_mov_b32 s4, SCRATCH_RSRC_DWORD0
; GCN-DAG: s_mov_b32 s5, SCRATCH_RSRC_DWORD1
; GCN-DAG: s_mov_b32 s6, -1

; CI-DAG: s_mov_b32 s7, 0xe8f000
; VI-DAG: s_mov_b32 s7, 0xe80000
; GFX9-DAG: s_mov_b32 s7, 0xe00000

; GCN: s_add_u32 s4, s4, s2
; GCN: s_addc_u32 s5, s5, 0

; GCN: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, s[4:7], 0 offen
; GCN: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, s[4:7], 0 offen

; ALL: ; ScratchSize: 32772
define amdgpu_ps void @large_alloca_pixel_shader_inreg(i32 inreg %x, i32 inreg %y) #0 {
  %large = alloca [8192 x i32], align 4, addrspace(5)
  %gep = getelementptr [8192 x i32], ptr addrspace(5) %large, i32 0, i32 8191
  store volatile i32 %x, ptr addrspace(5) %gep
  %gep1 = getelementptr [8192 x i32], ptr addrspace(5) %large, i32 0, i32 %y
  %val = load volatile i32, ptr addrspace(5) %gep1
  store volatile i32 %val, ptr addrspace(1) undef
  ret void
}

attributes #0 = { nounwind  }
