; RUN: llc -global-isel=0 -amdgpu-fixed-function-abi=0 -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefixes=GCN,SDAG %s
; RUN: llc -global-isel=1 -amdgpu-fixed-function-abi=1 -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefixes=GCN,GISEL %s

; FIXME: Emitting unnecessary flat_scratch setup

; GCN-LABEL: {{^}}test_call_undef:
; SDAG: s_mov_b32 flat_scratch_lo, s13
; SDAG: s_add_i32 s12, s12, s17
; SDAG: s_lshr_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_call_undef() #0 {
  %val = call i32 undef(i32 1)
  %op = add i32 %val, 1
  store volatile i32 %op, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_tail_call_undef:
; SDAG: s_waitcnt
; SDAG-NEXT: .Lfunc_end

; GISEL: s_setpc_b64 s{{\[[0-9]+:[0-9]+\]}}
define i32 @test_tail_call_undef() #0 {
  %call = tail call i32 undef(i32 1)
  ret i32 %call
}

; GCN-LABEL: {{^}}test_call_null:
; SDAG: s_mov_b32 flat_scratch_lo, s13
; SDAG: s_add_i32 s12, s12, s17
; SDAG: s_lshr_b32

; GISEL: s_swappc_b64 s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @test_call_null() #0 {
  %val = call i32 null(i32 1)
  %op = add i32 %val, 1
  store volatile i32 %op, i32 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}test_tail_call_null:
; SDAG: s_waitcnt
; SDAG-NEXT: .Lfunc_end

; GISEL: s_setpc_b64 s{{\[[0-9]+:[0-9]+\]$}}
define i32 @test_tail_call_null() #0 {
  %call = tail call i32 null(i32 1)
  ret i32 %call
}
