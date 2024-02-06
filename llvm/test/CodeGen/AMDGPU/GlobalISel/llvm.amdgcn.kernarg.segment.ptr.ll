; RUN: llc -global-isel -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V4,HSA,ALL %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=hawaii -mattr=+flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V4,OS-MESA3D,ALL %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-unknown -mcpu=hawaii -mattr=+flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=OS-UNKNOWN,ALL %s

; ALL-LABEL: {{^}}test:
; OS-MESA3D: enable_sgpr_kernarg_segment_ptr = 1
; CO-V4: s_load_dword s{{[0-9]+}}, s[4:5], 0xa

; OS-UNKNOWN: s_load_dword s{{[0-9]+}}, s[0:1], 0xa

; HSA: .amdhsa_kernarg_size 8
; HSA: .amdhsa_user_sgpr_kernarg_segment_ptr 1
define amdgpu_kernel void @test(ptr addrspace(1) %out) #1 {
  %kernarg.segment.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %gep = getelementptr i32, ptr addrspace(4) %kernarg.segment.ptr, i64 10
  %value = load i32, ptr addrspace(4) %gep
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit:
; OS-MESA3D: kernarg_segment_byte_size = 24
; OS-MESA3D: kernarg_segment_alignment = 4

; 10 + 9 (36 prepended implicit bytes) + 2(out pointer) = 21 = 0x15

; OS-UNKNOWN: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x15

; HSA:        .amdhsa_kernarg_size 8
define amdgpu_kernel void @test_implicit(ptr addrspace(1) %out) #1 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr i32, ptr addrspace(4) %implicitarg.ptr, i64 10
  %value = load i32, ptr addrspace(4) %gep
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit_alignment:
; OS-MESA3D: kernarg_segment_byte_size = 28
; OS-MESA3D: kernarg_segment_alignment = 4

; OS-UNKNOWN: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xc
; HSA: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x4
; OS-MESA3D: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x3
; ALL: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[VAL]]
; ALL: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[V_VAL]]

; HSA:        .amdhsa_kernarg_size 12
define amdgpu_kernel void @test_implicit_alignment(ptr addrspace(1) %out, <2 x i8> %in) #1 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %val = load i32, ptr addrspace(4) %implicitarg.ptr
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment
; OS-MESA3D: kernarg_segment_byte_size = 28
; OS-MESA3D: kernarg_segment_alignment = 4

; OS-UNKNOWN: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xc
; HSA: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x4
; OS-MESA3D: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x3
; ALL: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[VAL]]
; ALL: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[V_VAL]]

; HSA:        .amdhsa_kernarg_size 64
define amdgpu_kernel void @opencl_test_implicit_alignment(ptr addrspace(1) %out, <2 x i8> %in) #2 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %val = load i32, ptr addrspace(4) %implicitarg.ptr
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}test_no_kernargs:
; OS-MESA3D: enable_sgpr_kernarg_segment_ptr = 0
; OS-MESA3D: kernarg_segment_byte_size = 0
; OS-MESA3D: kernarg_segment_alignment = 4

; HSA: s_mov_b64 [[OFFSET_NULL:s\[[0-9]+:[0-9]+\]]], 40{{$}}
; HSA: s_load_dword s{{[0-9]+}}, [[OFFSET_NULL]]

; HSA: .amdhsa_kernarg_size 0
; HSA: .amdhsa_user_sgpr_kernarg_segment_ptr 0
define amdgpu_kernel void @test_no_kernargs() #1 {
  %kernarg.segment.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %gep = getelementptr i32, ptr addrspace(4) %kernarg.segment.ptr, i64 10
  %value = load i32, ptr addrspace(4) %gep
  store volatile i32 %value, ptr addrspace(1) undef
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment_no_explicit_kernargs:
; OS-MESA3D: kernarg_segment_byte_size = 16
; OS-MESA3D: kernarg_segment_alignment = 4
; HSA:        .amdhsa_kernarg_size 48
define amdgpu_kernel void @opencl_test_implicit_alignment_no_explicit_kernargs() #2 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %val = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment_no_explicit_kernargs_round_up:
; OS-MESA3D: kernarg_segment_byte_size = 16
; OS-MESA3D: kernarg_segment_alignment = 4
; HSA:        .amdhsa_kernarg_size 40
define amdgpu_kernel void @opencl_test_implicit_alignment_no_explicit_kernargs_round_up() #3 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %val = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

; ALL-LABEL: {{^}}func_kernarg_segment_ptr:
; ALL: v_mov_b32_e32 v0, 0{{$}}
; ALL: v_mov_b32_e32 v1, 0{{$}}
define ptr addrspace(4) @func_kernarg_segment_ptr() {
  %ptr = call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  ret ptr addrspace(4) %ptr
}

declare ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #0
declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "amdgpu-implicitarg-num-bytes"="0" }
attributes #2 = { nounwind "amdgpu-implicitarg-num-bytes"="48" }
attributes #3 = { nounwind "amdgpu-implicitarg-num-bytes"="38" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
