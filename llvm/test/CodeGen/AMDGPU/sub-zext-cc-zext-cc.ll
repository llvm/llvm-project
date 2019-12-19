; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs <%s | FileCheck -check-prefixes=GCN %s
;
; This test checks that we have the correct fold for zext(cc1) - zext(cc2).
;
; GCN-LABEL: sub_zext_zext:
; GCN: ds_read_b32 [[VAL:v[0-9]+]],
; GCN-DAG: v_cmp_lt_f32{{.*}} [[CC1:s\[[0-9]+:[0-9]+\]]], 0, [[VAL]]
; GCN-DAG: v_cmp_gt_f32{{.*}} vcc, 0, [[VAL]]
; GCN: v_cndmask_{{.*}} [[ZEXTCC1:v[0-9]+]], 0, 1, [[CC1]]
; GCN: v_subbrev{{.*}} {{v[0-9]+}}, vcc, 0, [[ZEXTCC1]], vcc
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

  %t519 = load float, float addrspace(3)* null

  %t524 = fcmp ogt float %t519, 0.000000e+00
  %t525 = fcmp olt float %t519, 0.000000e+00
  %t526 = zext i1 %t524 to i32
  %t527 = zext i1 %t525 to i32
  %t528 = sub nsw i32 %t526, %t527
  %t529 = sitofp i32 %t528 to float
  ret float %t529
}

