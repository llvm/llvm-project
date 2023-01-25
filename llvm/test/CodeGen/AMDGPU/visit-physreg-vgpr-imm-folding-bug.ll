; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; Make sure the return value of the first call is not overwritten with
; a constant before the fadd use.

; CHECK-LABEL: vgpr_multi_use_imm_fold:
; CHECK: v_mov_b32_e32 v0, 0{{$}}
; CHECK: v_mov_b32_e32 v1, 2.0{{$}}
; CHECK:    s_swappc_b64
; CHECK-NEXT: v_add_f64 v[0:1], v[0:1], 0
; CHECK:    s_swappc_b64
define amdgpu_kernel void @vgpr_multi_use_imm_fold() {
entry:
  store double 0.0, ptr addrspace(1) undef, align 8
  %call0 = tail call fastcc double @__ocml_log_f64(double 2.0)
  %op = fadd double %call0, 0.0
  %call1 = tail call fastcc double @__ocml_sqrt_f64(double %op)
  ret void
}

declare hidden fastcc double @__ocml_log_f64(double)
declare hidden fastcc double @__ocml_sqrt_f64(double)
