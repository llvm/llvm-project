; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -amdgpu-spill-sgpr-to-vgpr=true < %s | FileCheck -check-prefixes=GCN,SPILL-TO-VGPR %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -amdgpu-spill-sgpr-to-vgpr=false < %s | FileCheck -check-prefixes=GCN,NO-SPILL-TO-VGPR %s

; Check frame setup where SGPR spills to VGPRs are disabled or enabled.

declare hidden void @external_void_func_void() #0

; GCN-LABEL: {{^}}callee_with_stack_and_call:
; SPILL-TO-VGPR:      buffer_store_dword v40, off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; SPILL-TO-VGPR:      v_writelane_b32 v40, s33, 2
; SPILL-TO-VGPR:      v_writelane_b32 v40, s30, 0
; SPILL-TO-VGPR:      v_writelane_b32 v40, s31, 1
; NO-SPILL-TO-VGPR:   v_mov_b32_e32 v0, s33
; NO-SPILL-TO-VGPR:   buffer_store_dword v0, off, s[0:3], s32 offset:12 ; 4-byte Folded Spill
; NO-SPILL-TO-VGPR:   v_writelane_b32 v1, s30, 0
; NO-SPILL-TO-VGPR:   v_writelane_b32 v1, s31, 1
; NO-SPILL-TO-VGPR:   buffer_store_dword v1, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill

; GCN:                s_swappc_b64 s[30:31], s[4:5]

; SPILL-TO-VGPR:      v_readlane_b32 s4, v40, 0
; SPILL-TO-VGPR:      v_readlane_b32 s5, v40, 1
; NO-SPILL-TO-VGPR:   buffer_load_dword v1, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
; NO-SPILL-TO-VGPR:   v_readlane_b32 s4, v1, 0
; NO-SPILL-TO-VGPR:   v_readlane_b32 s5, v1, 1

; SPILL-TO-VGPR:      v_readlane_b32 s33, v40, 2
; NO-SPILL-TO-VGPR:   buffer_load_dword v0, off, s[0:3], s32 offset:12 ; 4-byte Folded Reload
; NO-SPILL-TO-VGPR:   v_readfirstlane_b32 s33, v0
define void @callee_with_stack_and_call() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void @external_void_func_void()
  ret void
}

attributes #0 = { nounwind }
