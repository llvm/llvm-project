; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s

; FIXME: Emitting unnecessary flat_scratch setup

; GCN-LABEL: {{^}}test_call_undef:
; GCN: s_mov_b32 s8, s7
; GCN: s_mov_b32 flat_scratch_lo, s5
; GCN: s_add_u32 s4, s4, s8
; GCN: s_lshr_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_call_undef() #0 {
  %val = call i32 undef(i32 1)
  %op = add i32 %val, 1
  store volatile i32 %op, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_tail_call_undef:
; GCN: s_waitcnt
; GCN-NEXT: .Lfunc_end
define i32 @test_tail_call_undef() #0 {
  %call = tail call i32 undef(i32 1)
  ret i32 %call
}

; GCN-LABEL: {{^}}test_call_null:
; GCN: s_mov_b32 s8, s7
; GCN: s_mov_b32 flat_scratch_lo, s5
; GCN: s_add_u32 s4, s4, s8
; GCN: s_lshr_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_call_null() #0 {
  %val = call i32 null(i32 1)
  %op = add i32 %val, 1
  store volatile i32 %op, i32 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}test_tail_call_null:
; GCN: s_waitcnt
; GCN-NEXT: .Lfunc_end
define i32 @test_tail_call_null() #0 {
  %call = tail call i32 null(i32 1)
  ret i32 %call
}
