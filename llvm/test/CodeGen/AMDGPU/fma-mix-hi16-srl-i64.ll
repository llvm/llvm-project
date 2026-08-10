; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -O0 < %s | FileCheck %s

; Test that isExtractHiElt in SelectVOP3PMadMixModsImpl does not fold a 64-bit
; source into V_FMA_MIX_F32 which expects a 32-bit source operand.
; At -O0 the DAG combiner does not decompose i64 to i32, so the pattern
; trunc(srl(i64, 16)) survives to isel. isExtractHiElt must reject it because
; V_FMA_MIX_F32 reads hi16 from a 32-bit register via op_sel.

; CHECK-LABEL: _amdgpu_cs_main:
; CHECK: s_load_b64
; CHECK: v_fma_mix_f32
; CHECK: global_store_b64
; CHECK: ; return to shader part epilog
define amdgpu_cs float @_amdgpu_cs_main(ptr addrspace(4) inreg %p, ptr addrspace(1) %out) {
entry:
  %vec = load <4 x half>, ptr addrspace(4) %p
  %0 = extractelement <4 x half> %vec, i64 1
  %ch1.ext = fpext half %0 to float
  %mul = fmul float %ch1.ext, 0.000000e+00
  store <4 x half> %vec, ptr addrspace(1) %out
  ret float %mul
}
