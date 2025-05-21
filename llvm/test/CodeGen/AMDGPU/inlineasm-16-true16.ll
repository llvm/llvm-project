; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 -verify-machineinstrs < %s 2>&1 | FileCheck -enable-var-scope -check-prefixes=GFX11 %s

; GFX11-LABEL: {{^}}s_input_output_i16:
; GFX11: s_mov_b32 s[[REG:[0-9]+]], -1
; GFX11: ; use s[[REG]]
define amdgpu_kernel void @s_input_output_i16() #0 {
  %v = tail call i16 asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(i16 %v) #0
  ret void
}

; GFX11-LABEL: {{^}}v_input_output_i16:
; GFX11: v_mov_b16 v[[REG:[0-9]+.(l|h)]], -1
; GFX11: ; use v[[REG]]
define amdgpu_kernel void @v_input_output_i16() #0 {
  %v = tail call i16 asm sideeffect "v_mov_b16 $0, -1", "=v"() #0
  tail call void asm sideeffect "; use $0", "v"(i16 %v)
  ret void
}

; GFX11-LABEL: {{^}}s_input_output_f16:
; GFX11: s_mov_b32 s[[REG:[0-9]+]], -1
; GFX11: ; use s[[REG]]
define amdgpu_kernel void @s_input_output_f16() #0 {
  %v = tail call half asm sideeffect "s_mov_b32 $0, -1", "=s"() #0
  tail call void asm sideeffect "; use $0", "s"(half %v)
  ret void
}

; GFX11-LABEL: {{^}}v_input_output_f16:
; GFX11: v_mov_b16 v[[REG:[0-9]+.(l|h)]], -1
; GFX11: ; use v[[REG]]
define amdgpu_kernel void @v_input_output_f16() #0 {
  %v = tail call half asm sideeffect "v_mov_b16 $0, -1", "=v"() #0
  tail call void asm sideeffect "; use $0", "v"(half %v)
  ret void
}

; GFX11-LABEL: {{^}}i16_imm_input_phys_vgpr:
; GFX11: v_mov_b16_e32 v0.l, -1
; GFX11: ; use v0
define amdgpu_kernel void @i16_imm_input_phys_vgpr() {
entry:
  call void asm sideeffect "; use $0 ", "{v0.l}"(i16 65535)
  ret void
}

attributes #0 = { nounwind }
