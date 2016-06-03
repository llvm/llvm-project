; RUN: llc -march=amdgcn -mcpu=bonaire < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=ALL %s
; RUN: llc -march=amdgcn -mcpu=carrizo < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=ALL %s

; ALL-LABEL: {{^}}large_alloca_pixel_shader:
; GCN: s_mov_b32 s8, SCRATCH_RSRC_DWORD0
; GCN: s_mov_b32 s9, SCRATCH_RSRC_DWORD1
; GCN: s_mov_b32 s10, -1
; CI: s_mov_b32 s11, 0x98f000
; VI: s_mov_b32 s11, 0x980000

; GCN: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, s[8:11], s1 offen
; GCN: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, s[8:11], s1 offen

; ALL: ; ScratchSize: 32772
define void @large_alloca_pixel_shader(i32 %x, i32 %y) #1 {
  %large = alloca [8192 x i32], align 4
  %gep = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 8191
  store volatile i32 %x, i32* %gep
  %gep1 = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 %y
  %val = load volatile i32, i32* %gep1
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; ALL-LABEL: {{^}}large_alloca_pixel_shader_inreg:
; GCN: s_mov_b32 s8, SCRATCH_RSRC_DWORD0
; GCN: s_mov_b32 s9, SCRATCH_RSRC_DWORD1
; GCN: s_mov_b32 s10, -1
; CI: s_mov_b32 s11, 0x98f000
; VI: s_mov_b32 s11, 0x980000

; GCN: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, s[8:11], s1 offen
; GCN: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, s[8:11], s1 offen

; ALL: ; ScratchSize: 32772
define void @large_alloca_pixel_shader_inreg(i32 inreg %x, i32 inreg %y) #1 {
  %large = alloca [8192 x i32], align 4
  %gep = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 8191
  store volatile i32 %x, i32* %gep
  %gep1 = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 %y
  %val = load volatile i32, i32* %gep1
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind  }
attributes #1 = { nounwind "ShaderType"="0" }
