; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; --------------------------------------------------------------------------------
; Don't fold if fneg can fold into the source
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}select_fneg_posk_src_rcp_legacy_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_rcp_legacy_f32_e32 [[RCP:v[0-9]+]], [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -2.0, [[RCP]], vcc
; GCN: v_xor_b32_e32 [[NEG_SELECT:v[0-9]+]], 0x80000000, [[SELECT]]
; GCN-NEXT: buffer_store_dword [[NEG_SELECT]]
define amdgpu_kernel void @select_fneg_posk_src_rcp_legacy_f32(i32 %c) #2 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %rcp = call float @llvm.amdgcn.rcp.legacy(float %x)
  %fneg = fsub float -0.0, %rcp
  %select = select i1 %cmp, float %fneg, float 2.0
  store volatile float %select, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}select_fneg_posk_src_mul_legacy_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_mul_legacy_f32_e32 [[MUL:v[0-9]+]], 4.0, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -2.0, [[MUL]], vcc
; GCN: v_xor_b32_e32 [[NEG_SELECT:v[0-9]+]], 0x80000000, [[SELECT]]
; GCN-NEXT: buffer_store_dword [[NEG_SELECT]]
define amdgpu_kernel void @select_fneg_posk_src_mul_legacy_f32(i32 %c) #2 {
  %x = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %mul = call float @llvm.amdgcn.fmul.legacy(float %x, float 4.0)
  %fneg = fsub float -0.0, %mul
  %select = select i1 %cmp, float %fneg, float 2.0
  store volatile float %select, float addrspace(1)* undef
  ret void
}

declare float @llvm.amdgcn.rcp.legacy(float) #1
declare float @llvm.amdgcn.fmul.legacy(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
