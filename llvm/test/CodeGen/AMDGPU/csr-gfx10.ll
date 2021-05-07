; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -o - %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -o - %s | FileCheck -check-prefix=GFX10PLUS %s

; Make sure new higher SGPRs are callee saved
; GFX10PLUS-LABEL: {{^}}callee_new_sgprs:
; GFX10PLUS: v_writelane_b32 v0, s104, 0
; GFX10PLUS-DAG: v_writelane_b32 v0, s105, 1
; GFX10PLUS-DAG: ; clobber s104
; GFX10PLUS: ; clobber s105
; GFX10PLUS: v_readlane_b32 s105, v0, 1
; GFX10PLUS: v_readlane_b32 s104, v0, 0
define void @callee_new_sgprs() {
  call void asm sideeffect "; clobber s104", "~{s104}"()
  call void asm sideeffect "; clobber s105", "~{s105}"()
  ret void
}
