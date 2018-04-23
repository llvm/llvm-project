; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -stress-regalloc=1 < %s | FileCheck -check-prefix=GCN %s

; For the CSR copy of s5, it may be possible to see it in
; storeRegToStackSlot.

; GCN-LABEL: {{^}}spill_csr_s5_copy:
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:8 ; 4-byte Folded Spill
; GCN: v_writelane_b32 v32, s5, 2
; GCN: s_swappc_b64
; GCN: v_readlane_b32 s5, v32, 2
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 9
; GCN: buffer_store_dword [[K]], off, s[0:3], s5 offset:4
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:8 ; 4-byte Folded Reload
; GCN: s_setpc_b64
define void @spill_csr_s5_copy() #0 {
bb:
  %alloca = alloca i32, addrspace(5)
  %tmp = tail call i64 @func() #1
  %tmp1 = getelementptr inbounds i32, i32 addrspace(1)* null, i64 %tmp
  %tmp2 = load i32, i32 addrspace(1)* %tmp1, align 4
  %tmp3 = zext i32 %tmp2 to i64
  store volatile i32 9, i32 addrspace(5)* %alloca
  ret void
}

declare i64 @func()

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
