; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; Make sure fdiv is promoted to f32.

; GCN-LABEL: {{^}}fdiv_f16
; GCN:     v_cvt_f32_f16
; GCN:     v_cvt_f32_f16
; GCN:     v_div_scale_f32
; GCN-DAG: v_div_scale_f32
; GCN-DAG: v_rcp_f32
; GCN:     v_fma_f32
; GCN:     v_fma_f32
; GCN:     v_mul_f32
; GCN:     v_fma_f32
; GCN:     v_fma_f32
; GCN:     v_fma_f32
; GCN:     v_div_fmas_f32
; GCN:     v_div_fixup_f32
; GCN:     v_cvt_f16_f32
define void @fdiv_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fdiv half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}
