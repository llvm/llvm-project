; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=SI,FUNC,GFX7 %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=SI,FUNC,GFX8 %s

; On Southern Islands GPUs the local address space(3) uses 32-bit pointers and
; the global address space(1) uses 64-bit pointers.  These tests check to make sure
; the correct pointer size is used for the local address space.

; The e{{32|64}} suffix on the instructions refers to the encoding size and not
; the size of the operands.  The operand size is denoted in the instruction name.
; Instructions with B32, U32, and I32 in their name take 32-bit operands, while
; instructions with B64, U64, and I64 take 64-bit operands.

; FUNC-LABEL: {{^}}local_address_load:
; SI: v_mov_b32_e{{32|64}} [[PTR:v[0-9]]]
; SI: ds_read_b32 v{{[0-9]+}}, [[PTR]]
define amdgpu_kernel void @local_address_load(ptr addrspace(1) %out, ptr addrspace(3) %in) {
entry:
  %0 = load i32, ptr addrspace(3) %in
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_address_gep:
; SI: s_add_i32 [[SPTR:s[0-9]]]
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: ds_read_b32 [[VPTR]]
define amdgpu_kernel void @local_address_gep(ptr addrspace(1) %out, ptr addrspace(3) %in, i32 %offset) {
entry:
  %0 = getelementptr i32, ptr addrspace(3) %in, i32 %offset
  %1 = load i32, ptr addrspace(3) %0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_address_gep_const_offset:
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], s{{[0-9]+}}
; SI: ds_read_b32 v{{[0-9]+}}, [[VPTR]] offset:4
define amdgpu_kernel void @local_address_gep_const_offset(ptr addrspace(1) %out, ptr addrspace(3) %in) {
entry:
  %0 = getelementptr i32, ptr addrspace(3) %in, i32 1
  %1 = load i32, ptr addrspace(3) %0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; Offset too large, can't fold into 16-bit immediate offset.
; FUNC-LABEL: {{^}}local_address_gep_large_const_offset:
; SI: s_add_i32 [[SPTR:s[0-9]]], s{{[0-9]+}}, 0x10004
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: ds_read_b32 [[VPTR]]
define amdgpu_kernel void @local_address_gep_large_const_offset(ptr addrspace(1) %out, ptr addrspace(3) %in) {
entry:
  %0 = getelementptr i32, ptr addrspace(3) %in, i32 16385
  %1 = load i32, ptr addrspace(3) %0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}null_32bit_lds_ptr:
; GFX7 v_cmp_ne_u32
; GFX7: s_cselect_b32
; GFX8: s_cmp_lg_u32
; GFX8-NOT: v_cmp_ne_u32
; GFX8: s_cselect_b32
define amdgpu_kernel void @null_32bit_lds_ptr(ptr addrspace(1) %out, ptr addrspace(3) %lds) nounwind {
  %cmp = icmp ne ptr addrspace(3) %lds, null
  %x = select i1 %cmp, i32 123, i32 456
  store i32 %x, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}mul_32bit_ptr:
; SI: s_mul_i32
; SI-NEXT: s_add_i32
; SI: ds_read_b32
define amdgpu_kernel void @mul_32bit_ptr(ptr addrspace(1) %out, ptr addrspace(3) %lds, i32 %tid) {
  %ptr = getelementptr [3 x float], ptr addrspace(3) %lds, i32 %tid, i32 0
  %val = load float, ptr addrspace(3) %ptr
  store float %val, ptr addrspace(1) %out
  ret void
}

@g_lds = addrspace(3) global float undef, align 4

; FUNC-LABEL: {{^}}infer_ptr_alignment_global_offset:
; SI: v_mov_b32_e32 [[PTR:v[0-9]+]], 0{{$}}
; SI: ds_read_b32 v{{[0-9]+}}, [[PTR]]
define amdgpu_kernel void @infer_ptr_alignment_global_offset(ptr addrspace(1) %out, i32 %tid) {
  %val = load float, ptr addrspace(3) @g_lds
  store float %val, ptr addrspace(1) %out
  ret void
}


@ptr = addrspace(3) global ptr addrspace(3) undef
@dst = addrspace(3) global [16383 x i32] undef

; FUNC-LABEL: {{^}}global_ptr:
; SI: ds_write_b32
define amdgpu_kernel void @global_ptr() nounwind {
  store ptr addrspace(3) getelementptr ([16383 x i32], ptr addrspace(3) @dst, i32 0, i32 16), ptr addrspace(3) @ptr
  ret void
}

; FUNC-LABEL: {{^}}local_address_store:
; SI: ds_write_b32
define amdgpu_kernel void @local_address_store(ptr addrspace(3) %out, i32 %val) {
  store i32 %val, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_address_gep_store:
; SI: s_add_i32 [[SADDR:s[0-9]+]],
; SI: v_mov_b32_e32 [[ADDR:v[0-9]+]], [[SADDR]]
; SI: ds_write_b32 [[ADDR]], v{{[0-9]+}}
define amdgpu_kernel void @local_address_gep_store(ptr addrspace(3) %out, i32, i32 %val, i32 %offset) {
  %gep = getelementptr i32, ptr addrspace(3) %out, i32 %offset
  store i32 %val, ptr addrspace(3) %gep, align 4
  ret void
}

; FUNC-LABEL: {{^}}local_address_gep_const_offset_store:
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], s{{[0-9]+}}
; SI: v_mov_b32_e32 [[VAL:v[0-9]+]], s{{[0-9]+}}
; SI: ds_write_b32 [[VPTR]], [[VAL]] offset:4
define amdgpu_kernel void @local_address_gep_const_offset_store(ptr addrspace(3) %out, i32 %val) {
  %gep = getelementptr i32, ptr addrspace(3) %out, i32 1
  store i32 %val, ptr addrspace(3) %gep, align 4
  ret void
}

; Offset too large, can't fold into 16-bit immediate offset.
; FUNC-LABEL: {{^}}local_address_gep_large_const_offset_store:
; SI: s_add_i32 [[SPTR:s[0-9]]], s{{[0-9]+}}, 0x10004
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: ds_write_b32 [[VPTR]], v{{[0-9]+$}}
define amdgpu_kernel void @local_address_gep_large_const_offset_store(ptr addrspace(3) %out, i32 %val) {
  %gep = getelementptr i32, ptr addrspace(3) %out, i32 16385
  store i32 %val, ptr addrspace(3) %gep, align 4
  ret void
}
