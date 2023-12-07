; RUN: llc -O0 -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck %s

; FIXME: Merge into indirect-addressing-si.ll

; Make sure that TwoAddressInstructions keeps src0 as subregister sub0
; of the tied implicit use and def of the super register.

; CHECK-LABEL: {{^}}insert_wo_offset:
; CHECK: s_load_dword [[IN:s[0-9]+]]
; CHECK: s_mov_b32 m0, [[IN]]
; CHECK: v_movreld_b32_e32 v[[ELT0:[0-9]+]]
; CHECK: buffer_store_dwordx4
; CHECK: buffer_store_dwordx4
; CHECK: buffer_store_dwordx4
; CHECK: buffer_store_dwordx4
define amdgpu_kernel void @insert_wo_offset(ptr addrspace(1) %out, i32 %in) {
entry:
  %ins = insertelement <16 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0>, float 17.0, i32 %in
  store <16 x float> %ins, ptr addrspace(1) %out
  ret void
}

; Make sure we don't hit use of undefined register errors when expanding an
; extract with undef index.

; CHECK-LABEL: {{^}}extract_adjacent_blocks:
; CHECK: s_load_dword [[ARG:s[0-9]+]]
; CHECK: s_cmp_lg_u32
; CHECK: s_cbranch_scc1 [[BB4:.LBB[0-9]+_[0-9]+]]

; CHECK: buffer_load_dwordx4

; CHECK: s_branch [[ENDBB:.LBB[0-9]+_[0-9]+]]

; CHECK: [[BB4]]:
; CHECK: buffer_load_dwordx4

; CHECK: [[ENDBB]]:
; CHECK: buffer_store_dword
; CHECK: s_endpgm

define amdgpu_kernel void @extract_adjacent_blocks(i32 %arg) #0 {
bb:
  %tmp = icmp eq i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb4

bb1:
  %tmp2 = load volatile <4 x float>, ptr addrspace(1) undef
  %tmp3 = extractelement <4 x float> %tmp2, i32 undef
  call void asm sideeffect "; reg use $0", "v"(<4 x float> %tmp2) #0 ; Prevent block optimize out
  br label %bb7

bb4:
  %tmp5 = load volatile <4 x float>, ptr addrspace(1) undef
  %tmp6 = extractelement <4 x float> %tmp5, i32 undef
  call void asm sideeffect "; reg use $0", "v"(<4 x float> %tmp5) #0 ; Prevent block optimize out
  br label %bb7

bb7:
  %tmp8 = phi float [ %tmp3, %bb1 ], [ %tmp6, %bb4 ]
  store volatile float %tmp8, ptr addrspace(1) undef
  ret void
}
