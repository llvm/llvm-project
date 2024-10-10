; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs <%s | FileCheck -check-prefixes=GCN %s
;
; This test checks that we have the correct fold for zext(cc1) - zext(cc2).
;
; GCN-LABEL: sub_zext_zext:
; GCN: ds_read_b32 [[VAL:v[0-9]+]],
; GCN: v_cmp_lt_f32{{.*}} vcc, 0, [[VAL]]
; GCN: v_cmp_gt_f32{{.*}} s[0:1], 0, v0
; GCN: s_and_b64 s[2:3], vcc, exec
; GCN: s_cselect_b32 s2, 1, 0
; GCN: s_cmp_lg_u64 s[0:1], 0
; GCN: s_subb_u32 s0, s2, 0
; GCN: v_cvt_f32_i32_e32 v0, s0
;
; Before the reversion that this test is attached to, the compiler commuted
; the operands to the sub and used different logic to select the addc/subc
; instruction:
;    sub zext (setcc), x => addcarry 0, x, setcc
;    sub sext (setcc), x => subcarry 0, x, setcc
;
; ... but that is bogus. I believe it is not possible to fold those commuted
; patterns into any form of addcarry or subcarry.

define amdgpu_cs float @sub_zext_zext() {
.entry:

  %t519 = load float, ptr addrspace(3) null

  %t524 = fcmp ogt float %t519, 0.000000e+00
  %t525 = fcmp olt float %t519, 0.000000e+00
  %t526 = zext i1 %t524 to i32
  %t527 = zext i1 %t525 to i32
  %t528 = sub nsw i32 %t526, %t527
  %t529 = sitofp i32 %t528 to float
  ret float %t529
}

