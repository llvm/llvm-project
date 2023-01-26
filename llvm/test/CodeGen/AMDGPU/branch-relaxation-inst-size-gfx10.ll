; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs -amdgpu-s-branch-bits=4 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -amdgpu-s-branch-bits=4 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs -amdgpu-s-branch-bits=4 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10 %s

; Make sure the code size estimate for inline asm is 12-bytes per
; instruction, rather than 8 in previous generations.

; GCN-LABEL: {{^}}long_forward_branch_gfx10only:
; GFX9: s_cmp_eq_u32
; GFX9-NEXT: s_cbranch_scc1

; GFX10: s_cmp_eq_u32
; GFX10-NEXT: s_cbranch_scc0
; GFX10: s_getpc_b64
; GFX10: s_add_u32
; GFX10: s_addc_u32
; GFX10: s_setpc_b64
define amdgpu_kernel void @long_forward_branch_gfx10only(ptr addrspace(1) %arg, i32 %cnd) #0 {
bb0:
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2 ; +9 dword branch

bb2:
    ; Estimated as 40-bytes on gfx10 (requiring a long branch), but
    ; 16-bytes on gfx9 (allowing a short branch)
  call void asm sideeffect
   "v_nop_e64
    v_nop_e64", ""() #0
  br label %bb3

bb3:
  store volatile i32 %cnd, ptr addrspace(1) %arg
  ret void
}
