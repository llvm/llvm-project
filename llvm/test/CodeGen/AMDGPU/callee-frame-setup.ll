; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefix=GCN -check-prefix=CI %s

; GCN-LABEL: {{^}}callee_no_stack:
; GCN: ; BB#0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack() #0 {
  ret void
}

; Requires frame pointer for access to local regular object.

; GCN-LABEL: {{^}}callee_with_stack:
; GCN: ; BB#0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_mov_b32 s5, s32
; GCN-NEXT: v_mov_b32_e32 v0, 0{{$}}
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], s5 offset:4{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack() #0 {
  %alloca = alloca i32
  store volatile i32 0, i32* %alloca
  ret void
}

attributes #0 = { nounwind }
