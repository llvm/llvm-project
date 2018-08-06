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
; GCN-DAG: s_add_u32 s32, s32, 0x400{{$}}
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
; GCN: s_sub_u32 s32, s32, 0x400

; GCN: s_setpc_b64
define void @callee_no_stack_with_call() #0 {
  call void @external_void_func_void()
  ret void
}

declare void @external_void_func_void() #0

; Make sure if a CSR vgpr is used for SGPR spilling, it is saved and restored
; GCN-LABEL: {{^}}callee_func_sgpr_spill_no_calls:
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:4 ; 4-byte Folded Spill
; GCN: v_writelane_b32 v32
; GCN: ;;#ASMSTART
; GCN: v_readlane_b32 s{{[0-9]+}}, v32
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:4 ; 4-byte Folded Reload
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_func_sgpr_spill_no_calls(i32 %in) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"() #0
  call void asm sideeffect "", "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}"() #0
  call void asm sideeffect "", "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23}"() #0
  call void asm sideeffect "", "~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"() #0

  %wide.sgpr0 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr1 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr2 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr5 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr3 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr4 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0

  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr0) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr1) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr2) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr3) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %wide.sgpr4) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr5) #0
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "no-frame-pointer-elim"="true" }
