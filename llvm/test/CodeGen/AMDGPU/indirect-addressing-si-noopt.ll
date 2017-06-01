; RUN: llc -O0 -march=amdgcn -verify-machineinstrs < %s | FileCheck %s

; FIXME: Merge into indirect-addressing-si.ll

; Make sure that TwoAddressInstructions keeps src0 as subregister sub0
; of the tied implicit use and def of the super register.

; CHECK-LABEL: {{^}}insert_wo_offset:
; CHECK: s_load_dword [[IN:s[0-9]+]]
; CHECK: s_mov_b32 m0, [[IN]]
; CHECK: v_movreld_b32_e32 v[[ELT0:[0-9]+]]
; CHECK-NEXT: buffer_store_dwordx4 v{{\[}}[[ELT0]]:
define void @insert_wo_offset(<4 x float> addrspace(1)* %out, i32 %in) {
entry:
  %ins = insertelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, float 5.0, i32 %in
  store <4 x float> %ins, <4 x float> addrspace(1)* %out
  ret void
}

