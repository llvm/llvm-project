; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefix=GCN -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}callee_no_stack:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack() #0 {
  ret void
}

; GCN-LABEL: {{^}}callee_no_stack_no_fp_elim:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack_no_fp_elim() #1 {
  ret void
}

; Requires frame pointer for access to local regular object.

; GCN-LABEL: {{^}}callee_with_stack:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_mov_b32 s5, s32
; GCN-NEXT: v_mov_b32_e32 v0, 0{{$}}
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], s5 offset:4{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}callee_with_stack_and_call:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:8

; GCN-DAG: v_writelane_b32 v32, s33,
; GCN-DAG: v_writelane_b32 v32, s34,
; GCN-DAG: v_writelane_b32 v32, s35,
; GCN-DAG: s_add_u32 s32, s32, 0x300{{$}}
; GCN-DAG: v_mov_b32_e32 v0, 0{{$}}
; GCN-DAG: buffer_store_dword v0, off, s[0:3], s5 offset:4{{$}}
; GCN-DAG: s_mov_b32 s33, s5


; GCN: s_swappc_b64
; GCN: s_mov_b32 s5, s33
; GCN-DAG: v_readlane_b32 s35,
; GCN-DAG: v_readlane_b32 s34,
; GCN-DAG: v_readlane_b32 s33,
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:8
; GCN: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack_and_call() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void @external_void_func_void()
  ret void
}

; Should be able to copy incoming stack pointer directly to inner
; call's stack pointer argument.

; There is stack usage only because of the need to evict a VGPR for
; spilling CSR SGPRs.

; GCN-LABEL: {{^}}callee_no_stack_with_call:
; GCN: s_waitcnt
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:4
; GCN-DAG: v_writelane_b32 v32, s33, 0
; GCN-DAG: v_writelane_b32 v32, s34, 1
; GCN: s_mov_b32 s33, s5
; GCN: s_swappc_b64
; GCN: s_mov_b32 s5, s33

; GCN-DAG: v_readlane_b32 s34, v32, 1
; GCN-DAG: v_readlane_b32 s33, v32, 0
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:4
; GCN: s_sub_u32 s32, s32, 0x200

; GCN: s_setpc_b64
define void @callee_no_stack_with_call() #0 {
  call void @external_void_func_void()
  ret void
}

declare void @external_void_func_void() #0

attributes #0 = { nounwind }
attributes #1 = { nounwind "no-frame-pointer-elim"="true" }
