; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -o - %s | FileCheck -check-prefix=GFX10 %s

; Make sure new higher SGPRs are callee saved
; GFX10-LABEL: {{^}}callee_new_sgprs:
; GFX10: v_writelane_b32 v0, s104, 0
; GFX10-DAG: v_writelane_b32 v0, s105, 1
; GFX10-DAG: ; clobber s104
; GFX10: ; clobber s105
; GFX10: v_readlane_b32 s105, v0, 1
; GFX10: v_readlane_b32 s104, v0, 0
define void @callee_new_sgprs() {
  call void asm sideeffect "; clobber s104", "~{s104}"()
  call void asm sideeffect "; clobber s105", "~{s105}"()
  ret void
}
