; RUN: llc -global-isel=0 -mtriple=amdgpu12.50-- -o - %s | FileCheck %s --check-prefixes=CHECK,SDAG
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50-- -o - %s | FileCheck %s --check-prefixes=CHECK,GISEL

; A subtarget with more than 256 addressable VGPRs encodes a register number's
; high bits separately, with S_SET_VGPR_MSB. A whole-dword access folds its
; constant dword offset into the base register of each indexed move, so an
; access whose base reaches v256 or beyond needs those bits described -
; otherwise only the low eight bits are encoded and the move silently touches a
; register 256 lower than the one meant.
;
; The moves must therefore be in the named operand table, since that is how
; AMDGPULowerVGPREncoding finds the operands whose high bits it has to describe.
;
; Only SelectionDAG folds the offset into the base; GlobalISel folds it into the
; index instead and indexes from v0, so it cannot reach a high base this way and
; needs no mode change. Both are checked, because the difference is the reason
; this went unnoticed.

; The dword index is %i + 254, so a four-dword access spans v254, v255, v256 and
; v257 relative to M0.
; CHECK-LABEL: fold_across_256:
; SDAG:         v_movrels_b32_e32 v{{[0-9]+}}, v254
; SDAG-NEXT:    v_movrels_b32_e32 v{{[0-9]+}}, v255
; SDAG-NEXT:    s_set_vgpr_msb 1
; SDAG-NEXT:    v_movrels_b32_e32 v{{[0-9]+}}, v0
; SDAG-NEXT:    v_movrels_b32_e32 v{{[0-9]+}}, v1
; SDAG:         s_set_vgpr_msb 0x100
;
; GISEL:        s_lshl2_add_u32 s0, s0, 0x3f8
; GISEL:        v_movrels_b32_e32 v{{[0-9]+}}, v0
; GISEL-NOT:    s_set_vgpr_msb
define void @fold_across_256(ptr addrspace(1) %out, i32 inreg %i) {
  %s = shl nuw i32 %i, 2
  %a = add nuw i32 %s, 1016
  %p = inttoptr i32 %a to ptr addrspace(13)
  %v = load <4 x i32>, ptr addrspace(13) %p, align 16
  store <4 x i32> %v, ptr addrspace(1) %out, align 16
  ret void
}

; Entirely below 256, so neither path needs a mode change.
; CHECK-LABEL: fold_below_256:
; SDAG:         v_movrels_b32_e32 v{{[0-9]+}}, v4
; GISEL:        v_movrels_b32_e32 v{{[0-9]+}}, v0
; CHECK-NOT:    s_set_vgpr_msb
define void @fold_below_256(ptr addrspace(1) %out, i32 inreg %i) {
  %s = shl nuw i32 %i, 2
  %a = add nuw i32 %s, 16
  %p = inttoptr i32 %a to ptr addrspace(13)
  %v = load i32, ptr addrspace(13) %p, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
