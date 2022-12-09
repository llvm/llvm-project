; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,GFX11 %s

; GCN-LABEL: {{^}}smrd_imm_dlc:
; GFX10: s_buffer_load_dword s0, s[0:3], 0x0 dlc ; encoding: [0x00,0x40,0x20,0xf4,0x00,0x00,0x00,0xfa]
; GFX11: s_buffer_load_b32 s0, s[0:3], 0x0 dlc ; encoding: [0x00,0x20,0x20,0xf4,0x00,0x00,0x00,0xf8]
define amdgpu_ps float @smrd_imm_dlc(<4 x i32> inreg %desc) #0 {
main_body:
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 0, i32 4)
  ret float %r
}

; GCN-LABEL: {{^}}smrd_sgpr_dlc:
; GFX10: s_buffer_load_dword s0, s[0:3], s4 dlc ; encoding: [0x00,0x40,0x20,0xf4,0x00,0x00,0x00,0x08]
; GFX11: s_buffer_load_b32 s0, s[0:3], s4 dlc ; encoding: [0x00,0x20,0x20,0xf4,0x00,0x00,0x00,0x08]
define amdgpu_ps float @smrd_sgpr_dlc(<4 x i32> inreg %desc, i32 inreg %offset) #0 {
main_body:
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %offset, i32 4)
  ret float %r
}

; GCN-LABEL: {{^}}smrd_imm_glc_dlc:
; GFX10: s_buffer_load_dword s0, s[0:3], 0x0 glc dlc ; encoding: [0x00,0x40,0x21,0xf4,0x00,0x00,0x00,0xfa]
; GFX11: s_buffer_load_b32 s0, s[0:3], 0x0 glc dlc ; encoding: [0x00,0x60,0x20,0xf4,0x00,0x00,0x00,0xf8]
define amdgpu_ps float @smrd_imm_glc_dlc(<4 x i32> inreg %desc) #0 {
main_body:
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 0, i32 5)
  ret float %r
}

; GCN-LABEL: {{^}}smrd_sgpr_glc_dlc:
; GFX10: s_buffer_load_dword s0, s[0:3], s4 glc dlc ; encoding: [0x00,0x40,0x21,0xf4,0x00,0x00,0x00,0x08]
; GFX11: s_buffer_load_b32 s0, s[0:3], s4 glc dlc ; encoding: [0x00,0x60,0x20,0xf4,0x00,0x00,0x00,0x08]
define amdgpu_ps float @smrd_sgpr_glc_dlc(<4 x i32> inreg %desc, i32 inreg %offset) #0 {
main_body:
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %offset, i32 5)
  ret float %r
}

declare float @llvm.amdgcn.s.buffer.load.f32(<4 x i32>, i32, i32)

!0 = !{}
