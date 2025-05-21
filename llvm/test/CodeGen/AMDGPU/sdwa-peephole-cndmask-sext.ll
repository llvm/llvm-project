; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck %s
; XFAIL: *

; FIXME The sext modifier is turned into a neg modifier in the asm output

define i32 @test_select_on_sext_sdwa(i8 %x, i32 %y, i1 %cond)  {
; CHECK-LABEL: test_select_on_sext_sdwa:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_and_b32_e32 v2, 1, v2
; CHECK-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v2
; CHECK-NEXT:    v_mov_b32_e32 v2, 0
; CHECK-NEXT:    s_nop 0
; CHECK-NEXT:    v_cndmask_b32_sdwa v0, v2, sext(v0), vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
; CHECK-NEXT:    v_or_b32_e32 v0, v0, v1
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  %sext = sext i8 %x to i32
  %select = select i1 %cond, i32 %sext, i32 0
  %or = or i32 %select, %y
  ret i32 %or
}
