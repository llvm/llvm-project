; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Test that non-entry function frame indices are expanded properly to
; give an index relative to the scratch wave offset register

; Materialize into a mov. Make sure there isn't an unnecessary copy.
; GCN-LABEL: {{^}}func_mov_fi_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN: s_sub_u32 vcc_hi, s5, s4
; GCN-NEXT: s_lshr_b32 vcc_hi, vcc_hi, 6
; GCN-NEXT: v_add_i32_e64 v0, vcc, vcc_hi, 4
; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_mov_fi_i32() #0 {
  %alloca = alloca i32
  store volatile i32* %alloca, i32* addrspace(3)* undef
  ret void
}

; Materialize into an add of a constant offset from the FI.
; FIXME: Should be able to merge adds

; GCN-LABEL: {{^}}func_add_constant_to_fi_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN: s_sub_u32 s6, s5, s4
; GCN-NEXT: s_lshr_b32 s6, s6, 6
; GCN-NEXT: v_add_i32_e64 v0, s{{\[[0-9]+:[0-9]+\]}}, s6, 4
; GCN-NEXT: v_add_i32_e32 v0, vcc, 4, v0
; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_add_constant_to_fi_i32() #0 {
  %alloca = alloca [2 x i32], align 4
  %gep0 = getelementptr inbounds [2 x i32], [2 x i32]* %alloca, i32 0, i32 1
  store volatile i32* %gep0, i32* addrspace(3)* undef
  ret void
}

; A user the materialized frame index can't be meaningfully folded
; into.

; GCN-LABEL: {{^}}func_other_fi_user_i32:
; GCN: s_sub_u32 vcc_hi, s5, s4
; GCN-NEXT: s_lshr_b32 vcc_hi, vcc_hi, 6
; GCN-NEXT: v_add_i32_e64 v0, vcc, vcc_hi, 4
; GCN-NEXT: v_mul_lo_i32 v0, v0, 9
; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_other_fi_user_i32() #0 {
  %alloca = alloca [2 x i32], align 4
  %ptrtoint = ptrtoint [2 x i32]* %alloca to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, i32 addrspace(3)* undef
  ret void
}

; GCN-LABEL: {{^}}func_store_private_arg_i32_ptr:
; GCN: v_mov_b32_e32 v1, 15{{$}}
; GCN: buffer_store_dword v1, v0, s[0:3], s4 offen{{$}}
define void @func_store_private_arg_i32_ptr(i32* %ptr) #0 {
  store volatile i32 15, i32* %ptr
  ret void
}

; GCN-LABEL: {{^}}func_load_private_arg_i32_ptr:
; GCN: s_waitcnt
; GCN-NEXT: buffer_load_dword v0, v0, s[0:3], s4 offen{{$}}
define void @func_load_private_arg_i32_ptr(i32* %ptr) #0 {
  %val = load volatile i32, i32* %ptr
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr:
; GCN: s_waitcnt
; GCN-NEXT: s_sub_u32 s6, s5, s4
; GCN-NEXT: v_lshr_b32_e64 v0, s6, 6
; GCN-NEXT: v_add_i32_e32 v0, vcc, 4, v0
; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @void_func_byval_struct_i8_i32_ptr({ i8, i32 }* byval %arg0) #0 {
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %arg0, i32 0, i32 1
  %load1 = load i32, i32* %gep1
  store volatile i32* %gep1, i32* addrspace(3)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr_value:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: buffer_load_ubyte v0, off, s[0:3], s5
; GCN_NEXT: buffer_load_dword v1, off, s[0:3], s5 offset:4
define void @void_func_byval_struct_i8_i32_ptr_value({ i8, i32 }* byval %arg0) #0 {
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %arg0, i32 0, i32 1
  %load0 = load i8, i8* %gep0
  %load1 = load i32, i32* %gep1
  store volatile i8 %load0, i8 addrspace(3)* undef
  store volatile i32 %load1, i32 addrspace(3)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr_nonentry_block:
; GCN: s_sub_u32 s8, s5, s4
; GCN: v_lshr_b32_e64 v1, s8, 6
; GCN: s_and_saveexec_b64

; GCN: v_add_i32_e32 v0, vcc, 4, v1
; GCN: buffer_load_dword v1, v1, s[0:3], s4 offen offset:4
; GCN: ds_write_b32
define void @void_func_byval_struct_i8_i32_ptr_nonentry_block({ i8, i32 }* byval %arg0, i32 %arg2) #0 {
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %bb, label %ret

bb:
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %arg0, i32 0, i32 1
  %load1 = load volatile i32, i32* %gep1
  store volatile i32* %gep1, i32* addrspace(3)* undef
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind }
