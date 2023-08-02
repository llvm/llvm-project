; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -stress-regalloc=1 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}spill_csr_s5_copy:
; GCN: s_mov_b32 [[FP_SCRATCH_COPY:s[0-9]+]], s33
; GCN: s_xor_saveexec_b64
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], s33 offset:12 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, -1
; GCN-NEXT: buffer_store_dword v40, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec
; GCN: v_writelane_b32 v40, [[FP_SCRATCH_COPY]], 4
; GCN: s_swappc_b64

; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 9
; GCN: buffer_store_dword [[K]], off, s[0:3], s33{{$}}

; GCN: v_readlane_b32 [[FP_SCRATCH_COPY:s[0-9]+]], v40, 4
; GCN: s_xor_saveexec_b64
; GCN-NEXT: buffer_load_dword v0, off, s[0:3], s33 offset:12 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, -1
; GCN-NEXT: buffer_load_dword v40, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
; GCN: s_mov_b64 exec
; GCN: s_mov_b32 s33, [[FP_SCRATCH_COPY]]
; GCN: s_setpc_b64
define void @spill_csr_s5_copy() #0 {
bb:
  %alloca = alloca i32, addrspace(5)
  %tmp = tail call i64 @func() #1
  %tmp1 = getelementptr inbounds i32, ptr addrspace(1) null, i64 %tmp
  %tmp2 = load i32, ptr addrspace(1) %tmp1, align 4
  %tmp3 = zext i32 %tmp2 to i64
  store volatile i32 9, ptr addrspace(5) %alloca
  ret void
}

declare i64 @func() #0

attributes #0 = { nounwind "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
attributes #1 = { nounwind readnone }
