; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 2
; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 -disable-machine-sink=1 - < %s | FileCheck -check-prefix=GFX10 %s

define float @fold_abs_in_branch(float %arg1, float %arg2) {
; GFX10-LABEL: fold_abs_in_branch:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_add_f32_e32 v0, v0, v1
; GFX10-NEXT:    s_mov_b32 s4, exec_lo
; GFX10-NEXT:    v_add_f32_e32 v1, v0, v1
; GFX10-NEXT:    v_add_f32_e64 v0, |v1|, |v1|
; GFX10-NEXT:    v_cmpx_nlt_f32_e32 1.0, v0
; GFX10-NEXT:  ; %bb.1: ; %if
; GFX10-NEXT:    v_mul_f32_e64 v0, 0x3e4ccccd, |v1|
; GFX10-NEXT:  ; %bb.2: ; %exit
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_setpc_b64 s[30:31]
entry:
  %0 = fadd reassoc nnan nsz arcp contract afn float %arg1, %arg2
  %1 = fadd reassoc nnan nsz arcp contract afn float %0, %arg2
  %2 = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float %1)
  %3 = fmul reassoc nnan nsz arcp contract afn float %2, 2.000000e+00
  %4 = fcmp ule float %3, 1.000000e+00
  br i1 %4, label %if, label %exit

if:
  %if.3 = fmul reassoc nnan nsz arcp contract afn float %2, 0x3FC99999A0000000
  br label %exit

exit:
  %ret = phi float [ %3, %entry ], [ %if.3, %if ]
  ret float %ret
}

define float @fold_abs_in_branch_multiple_users(float %arg1, float %arg2) {
; GFX10-LABEL: fold_abs_in_branch_multiple_users:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_add_f32_e32 v0, v0, v1
; GFX10-NEXT:    s_mov_b32 s4, exec_lo
; GFX10-NEXT:    v_add_f32_e32 v0, v0, v1
; GFX10-NEXT:    v_add_f32_e64 v1, |v0|, |v0|
; GFX10-NEXT:    v_cmpx_nlt_f32_e32 1.0, v1
; GFX10-NEXT:  ; %bb.1: ; %if
; GFX10-NEXT:    v_mul_f32_e64 v1, 0x3e4ccccd, |v0|
; GFX10-NEXT:  ; %bb.2: ; %exit
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_add_f32_e64 v0, |v0|, 2.0
; GFX10-NEXT:    v_mul_f32_e32 v0, v0, v1
; GFX10-NEXT:    s_setpc_b64 s[30:31]
entry:
  %0 = fadd reassoc nnan nsz arcp contract afn float %arg1, %arg2
  %1 = fadd reassoc nnan nsz arcp contract afn float %0, %arg2
  %2 = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float %1)
  %3 = fmul reassoc nnan nsz arcp contract afn float %2, 2.000000e+00
  %4 = fcmp ule float %3, 1.000000e+00
  br i1 %4, label %if, label %exit

if:
  %if.3 = fmul reassoc nnan nsz arcp contract afn float %2, 0x3FC99999A0000000
  br label %exit

exit:
  %exit.phi = phi float [ %3, %entry ], [ %if.3, %if ]
  %ret.0 = fadd reassoc nnan nsz arcp contract afn float %2, 2.000000e+00
  %ret.1 = fmul float %ret.0, %exit.phi
  ret float %ret.1
}

define float @fold_abs_in_branch_undef(float %arg1, float %arg2) {
; GFX10-LABEL: fold_abs_in_branch_undef:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_add_f32_e64 v0, |s4|, |s4|
; GFX10-NEXT:    v_cmp_lt_f32_e32 vcc_lo, 1.0, v0
; GFX10-NEXT:    s_cbranch_vccnz .LBB2_2
; GFX10-NEXT:  ; %bb.1: ; %if
; GFX10-NEXT:    v_mul_f32_e64 v0, 0x3e4ccccd, |s4|
; GFX10-NEXT:  .LBB2_2: ; %exit
; GFX10-NEXT:    s_setpc_b64 s[30:31]
entry:
  %0 = fadd reassoc nnan nsz arcp contract afn float %arg1, %arg2
  %1 = fadd reassoc nnan nsz arcp contract afn float %0, %arg2
  %undef = freeze float poison
  %2 = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float %undef)
  %3 = fmul reassoc nnan nsz arcp contract afn float %2, 2.000000e+00
  %4 = fcmp ule float %3, 1.000000e+00
  br i1 %4, label %if, label %exit

if:
  %if.3 = fmul reassoc nnan nsz arcp contract afn float %2, 0x3FC99999A0000000
  br label %exit

exit:
  %ret = phi float [ %3, %entry ], [ %if.3, %if ]
  ret float %ret
}

define float @fold_abs_in_branch_poison(float %arg1, float %arg2) {
; GFX10-LABEL: fold_abs_in_branch_poison:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    s_setpc_b64 s[30:31]
entry:
  %0 = fadd reassoc nnan nsz arcp contract afn float %arg1, %arg2
  %1 = fadd reassoc nnan nsz arcp contract afn float %0, %arg2
  %2 = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float poison)
  %3 = fmul reassoc nnan nsz arcp contract afn float %2, 2.000000e+00
  %4 = fcmp ule float %3, 1.000000e+00
  br i1 %4, label %if, label %exit

if:
  %if.3 = fmul reassoc nnan nsz arcp contract afn float %2, 0x3FC99999A0000000
  br label %exit

exit:
  %ret = phi float [ %3, %entry ], [ %if.3, %if ]
  ret float %ret
}

define float @fold_abs_in_branch_fabs(float %arg1, float %arg2) {
; GFX10-LABEL: fold_abs_in_branch_fabs:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_add_f32_e32 v0, v0, v1
; GFX10-NEXT:    s_mov_b32 s4, exec_lo
; GFX10-NEXT:    v_add_f32_e32 v1, v0, v1
; GFX10-NEXT:    v_add_f32_e64 v0, |v1|, |v1|
; GFX10-NEXT:    v_cmpx_nlt_f32_e32 1.0, v0
; GFX10-NEXT:  ; %bb.1: ; %if
; GFX10-NEXT:    v_mul_f32_e64 v0, 0x3e4ccccd, |v1|
; GFX10-NEXT:  ; %bb.2: ; %exit
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_setpc_b64 s[30:31]
entry:
  %0 = fadd reassoc nnan nsz arcp contract afn float %arg1, %arg2
  %1 = fadd reassoc nnan nsz arcp contract afn float %0, %arg2
  %2 = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float %1)
  %3 = fmul reassoc nnan nsz arcp contract afn float %2, 2.000000e+00
  %4 = fcmp ule float %3, 1.000000e+00
  br i1 %4, label %if, label %exit

if:
  %if.fabs = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float %2)
  %if.3 = fmul reassoc nnan nsz arcp contract afn float %if.fabs, 0x3FC99999A0000000
  br label %exit

exit:
  %ret = phi float [ %3, %entry ], [ %if.3, %if ]
  ret float %ret
}

define float @fold_abs_in_branch_phi(float %arg1, float %arg2) {
; GFX10-LABEL: fold_abs_in_branch_phi:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_add_f32_e32 v0, v0, v1
; GFX10-NEXT:    s_mov_b32 s4, exec_lo
; GFX10-NEXT:    v_add_f32_e32 v0, v0, v1
; GFX10-NEXT:    v_add_f32_e64 v0, |v0|, |v0|
; GFX10-NEXT:    v_cmpx_nlt_f32_e32 1.0, v0
; GFX10-NEXT:    s_cbranch_execz .LBB5_3
; GFX10-NEXT:  ; %bb.1: ; %header.preheader
; GFX10-NEXT:    ; implicit-def: $vgpr0
; GFX10-NEXT:  .LBB5_2: ; %header
; GFX10-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX10-NEXT:    v_mul_f32_e32 v0, 0x40400000, v0
; GFX10-NEXT:    v_cmp_lt_f32_e32 vcc_lo, -1.0, v0
; GFX10-NEXT:    v_and_b32_e32 v0, 0x7fffffff, v0
; GFX10-NEXT:    s_cbranch_vccnz .LBB5_2
; GFX10-NEXT:  .LBB5_3: ; %Flow1
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_setpc_b64 s[30:31]
entry:
  %0 = fadd reassoc nnan nsz arcp contract afn float %arg1, %arg2
  %1 = fadd reassoc nnan nsz arcp contract afn float %0, %arg2
  %2 = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float %1)
  %3 = fmul reassoc nnan nsz arcp contract afn float %2, 2.000000e+00
  %4 = fcmp ule float %3, 1.000000e+00
  br i1 %4, label %header, label %exit

header:
  %h.fabs.phi = phi float [ poison, %entry ], [ %l.fabs, %l ]
  %h.fmul = fmul reassoc nnan nsz arcp contract afn float %h.fabs.phi, 2.000000e+00
  %l.1 = fmul reassoc nnan nsz arcp contract afn float %h.fabs.phi, 3.000000e+00
  br label %l

l:
  %l.e = fcmp ule float %l.1, -1.000000e+00
  %l.fabs = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float %l.1)
  br i1 %l.e, label %exit, label %header

exit:
  %ret = phi float [ %3, %entry ], [ %l.fabs, %l ]
  ret float %ret
}

define float @fold_neg_in_branch(float %arg1, float %arg2) {
; GFX10-LABEL: fold_neg_in_branch:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_add_f32_e32 v0, v0, v1
; GFX10-NEXT:    s_mov_b32 s4, exec_lo
; GFX10-NEXT:    v_add_f32_e32 v0, v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_cmpx_nlt_f32_e32 1.0, v0
; GFX10-NEXT:  ; %bb.1: ; %if
; GFX10-NEXT:    v_rcp_f32_e64 v1, -v0
; GFX10-NEXT:    v_mul_f32_e64 v1, |v0|, v1
; GFX10-NEXT:  ; %bb.2: ; %exit
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mul_f32_e64 v0, -v0, v1
; GFX10-NEXT:    s_setpc_b64 s[30:31]
entry:
  %0 = fadd reassoc nnan nsz arcp contract afn float %arg1, %arg2
  %1 = fadd reassoc nnan nsz arcp contract afn float %0, %arg2
  %2 = fneg reassoc nnan nsz arcp contract afn float %1
  %3 = fcmp ule float %1, 1.000000e+00
  br i1 %3, label %if, label %exit

if:
  %if.fabs = call reassoc nnan nsz arcp contract afn float @llvm.fabs.f32(float %1)
  %if.3 = fdiv reassoc nnan nsz arcp contract afn float %if.fabs, %2
  br label %exit

exit:
  %ret = phi float [ %1, %entry ], [ %if.3, %if ]
  %ret.2 = fmul reassoc nnan nsz arcp contract afn float %2, %ret
  ret float %ret.2
}

declare float @llvm.fabs.f32(float)
declare float @llvm.fmuladd.f32(float, float, float) #0
