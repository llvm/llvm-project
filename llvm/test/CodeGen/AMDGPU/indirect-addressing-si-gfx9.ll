; RUN: llc -amdgpu-scalarize-global-loads=false -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; indexing of vectors.

declare hidden void @foo()

; For functions with calls, we were not accounting for m0_lo16/m0_hi16
; uses on the BUNDLE created when expanding the insert register pseudo.
; GCN-LABEL: {{^}}insertelement_with_call:
; GCN: s_set_gpr_idx_on s{{[0-9]+}}, gpr_idx(DST)
; GCN-NEXT: v_mov_b32_e32 {{v[0-9]+}}, 8
; GCN-NEXT: s_set_gpr_idx_off
; GCN: s_swappc_b64
define amdgpu_kernel void @insertelement_with_call(ptr addrspace(1) %ptr, i32 %idx) #0 {
  %vec = load <16 x i32>, ptr addrspace(1) %ptr
  %i6 = insertelement <16 x i32> %vec, i32 8, i32 %idx
  call void @foo()
  store <16 x i32> %i6, ptr addrspace(1) null
  ret void
}

attributes #0 = { nounwind }
