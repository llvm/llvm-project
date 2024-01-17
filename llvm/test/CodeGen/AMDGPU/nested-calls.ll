; RUN: llc -mtriple=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; Test calls when called by other callable functions rather than
; kernels.

declare void @external_void_func_i32(i32) #0

; GCN-LABEL: {{^}}test_func_call_external_void_func_i32_imm:
; GCN: s_waitcnt

; Spill CSR VGPR used for SGPR spilling
; GCN: s_mov_b32 [[FP_SCRATCH_COPY:s[0-9]+]], s33
; GCN-NEXT: s_mov_b32 s33, s32
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_store_dword v40, off, s[0:3], s33 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-DAG: v_writelane_b32 v40, [[FP_SCRATCH_COPY]], 2
; GCN-DAG: v_writelane_b32 v40, s30, 0
; GCN-DAG: v_writelane_b32 v40, s31, 1

; GCN: s_swappc_b64

; GCN: v_readlane_b32 s30, v40, 0
; GCN: v_readlane_b32 s31, v40, 1

; GCN-NEXT: v_readlane_b32 [[FP_SCRATCH_COPY:s[0-9]+]], v40, 2
; GCN: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_load_dword v40, off, s[0:3], s33 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_addk_i32 s32, 0xfc00
; GCN-NEXT: s_mov_b32 s33, [[FP_SCRATCH_COPY]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64 s[30:31]
define void @test_func_call_external_void_func_i32_imm() #0 {
  call void @external_void_func_i32(i32 42)
  ret void
}

; GCN-LABEL: {{^}}test_func_call_external_void_func_i32_imm_stack_use:
; GCN: s_waitcnt
; GCN: s_mov_b32 s33, s32
; GCN-DAG: s_addk_i32 s32, 0x1400{{$}}
; GCN-DAG: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33 offset:
; GCN: s_swappc_b64
; GCN: s_addk_i32 s32, 0xec00{{$}}
; GCN: s_setpc_b64
define void @test_func_call_external_void_func_i32_imm_stack_use() #0 {
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep15 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 16
  store volatile i32 0, ptr addrspace(5) %alloca
  store volatile i32 0, ptr addrspace(5) %gep15
  call void @external_void_func_i32(i32 42)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind noinline }
