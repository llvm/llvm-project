; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1250 < %s | FileCheck -check-prefix=GFX1250 %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1200 < %s | FileCheck -check-prefix=GFX12 %s

; On GFX1250 (hasWaitXcnt), pure-uniform addresses should use VADDR to
; avoid s_wait_xcnt serialization when the RA reuses the SGPR pair.
; On GFX12, SADDR is still preferred for pure-uniform addresses.

define amdgpu_ps float @pure_uniform_global_load(ptr addrspace(1) inreg %sbase) {
; GFX1250-LABEL: pure_uniform_global_load:
; GFX1250:         global_load_b32 v0, v[0:1], off
;
; GFX12-LABEL: pure_uniform_global_load:
; GFX12:           global_load_b32 v0, v0, s[2:3]
  %val = load volatile float, ptr addrspace(1) %sbase
  ret float %val
}

; Verify that add(sgpr, zext(vgpr)) still selects SADDR on GFX1250.
define amdgpu_ps float @sgpr_plus_vgpr_offset(ptr addrspace(1) inreg %sbase, i32 %voffset) {
; GFX1250-LABEL: sgpr_plus_vgpr_offset:
; GFX1250:         global_load_b32 v0, v0, s[2:3]
;
; GFX12-LABEL: sgpr_plus_vgpr_offset:
; GFX12:           global_load_b32 v0, v0, s[2:3]
  %zext = zext i32 %voffset to i64
  %gep = getelementptr i8, ptr addrspace(1) %sbase, i64 %zext
  %val = load volatile float, ptr addrspace(1) %gep
  ret float %val
}
