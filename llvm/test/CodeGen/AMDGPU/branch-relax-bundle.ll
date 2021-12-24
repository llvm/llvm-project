; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -amdgpu-s-branch-bits=5 < %s | FileCheck -check-prefix=GCN %s

; Restrict maximum branch to between +15 and -16 dwords

; Instructions inside a bundle were collectively counted as
; 0-bytes. Make sure this is accounted for when estimating branch
; distances

; Bundle used for address in call sequence: 20 bytes
; s_getpc_b64
; s_add_u32
; s_addc_u32

; plus additional overhead
; s_setpc_b64
; and some register copies

declare void @func() #0

; GCN-LABEL: {{^}}bundle_size:
; GCN: s_cbranch_scc0 [[BB_EXPANSION:.LBB[0-9]+_[0-9]+]]
; GCN: s_getpc_b64
; GCN-NEXT: .Lpost_getpc{{[0-9]+}}:{{$}}
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64

; GCN: {{^}}[[BB_EXPANSION]]:
; GCN: s_getpc_b64
; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, func@
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, func@
; GCN: s_swappc_b64
define amdgpu_kernel void @bundle_size(i32 addrspace(1)* %arg, i32 %cnd) #0 {
bb:
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2 ; +8 dword branch

bb2:
  call void @func()
  call void asm sideeffect
  "v_nop_e64
   v_nop_e64
   v_nop_e64
   v_nop_e64
   v_nop_e64", ""() #0
  br label %bb3

bb3:
  store volatile i32 %cnd, i32 addrspace(1)* %arg
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
