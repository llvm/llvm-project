; RUN: llc -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V2,HSA,ALL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V2,OS-MESA3D,MESA,ALL %s
; RUN: llc -mtriple=amdgcn-mesa-unknown -verify-machineinstrs < %s | FileCheck -check-prefixes=OS-UNKNOWN,MESA,ALL %s

; ALL-LABEL: {{^}}test:
; CO-V2: enable_sgpr_kernarg_segment_ptr = 1
; HSA: kernarg_segment_byte_size = 8
; HSA: kernarg_segment_alignment = 4

; CO-V2: s_load_dword s{{[0-9]+}}, s[4:5], 0xa

; OS-UNKNOWN: s_load_dword s{{[0-9]+}}, s[0:1], 0xa
define amdgpu_kernel void @test(ptr addrspace(1) %out) #1 {
  %kernarg.segment.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %gep = getelementptr i32, ptr addrspace(4) %kernarg.segment.ptr, i64 10
  %value = load i32, ptr addrspace(4) %gep
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit:
; HSA: kernarg_segment_byte_size = 64
; OS-MESA3D: kernarg_segment_byte_size = 24
; CO-V2: kernarg_segment_alignment = 4

; 10 + 9 (36 prepended implicit bytes) + 2(out pointer) = 21 = 0x15
; OS-UNKNOWN: s_load_dword s{{[0-9]+}}, s[0:1], 0x15
define amdgpu_kernel void @test_implicit(ptr addrspace(1) %out) #1 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr i32, ptr addrspace(4) %implicitarg.ptr, i64 10
  %value = load i32, ptr addrspace(4) %gep
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit_alignment:
; HSA: kernarg_segment_byte_size = 72
; OS-MESA3D: kernarg_segment_byte_size = 28
; CO-V2: kernarg_segment_alignment = 4


; OS-UNKNOWN: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xc
; HSA: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x4
; OS-MESA3D: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x3
; ALL: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[VAL]]
; MESA: buffer_store_dword [[V_VAL]]
; HSA: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[V_VAL]]
define amdgpu_kernel void @test_implicit_alignment(ptr addrspace(1) %out, <2 x i8> %in) #1 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %val = load i32, ptr addrspace(4) %implicitarg.ptr
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment
; HSA: kernarg_segment_byte_size = 64
; OS-MESA3D: kernarg_segment_byte_size = 28
; CO-V2: kernarg_segment_alignment = 4


; OS-UNKNOWN: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xc
; HSA: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x4
; OS-MESA3D: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x3
; ALL: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[VAL]]
; MESA: buffer_store_dword [[V_VAL]]
; HSA: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[V_VAL]]
define amdgpu_kernel void @opencl_test_implicit_alignment(ptr addrspace(1) %out, <2 x i8> %in) #2 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %val = load i32, ptr addrspace(4) %implicitarg.ptr
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}test_no_kernargs:
; CO-V2: enable_sgpr_kernarg_segment_ptr = 0
; CO-V2: kernarg_segment_byte_size = 0

; CO-V2: kernarg_segment_alignment = 4

; HSA: s_mov_b64 [[NULL:s\[[0-9]+:[0-9]+\]]], 0{{$}}
; HSA: s_load_dword s{{[0-9]+}}, [[NULL]], 0xa{{$}}
define amdgpu_kernel void @test_no_kernargs() #1 {
  %kernarg.segment.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %gep = getelementptr i32, ptr addrspace(4) %kernarg.segment.ptr, i64 10
  %value = load i32, ptr addrspace(4) %gep
  store volatile i32 %value, ptr addrspace(1) undef
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment_no_explicit_kernargs:
; HSA: kernarg_segment_byte_size = 48
; OS-MESA3d: kernarg_segment_byte_size = 16
; CO-V2: kernarg_segment_alignment = 4
define amdgpu_kernel void @opencl_test_implicit_alignment_no_explicit_kernargs() #2 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %val = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment_no_explicit_kernargs_round_up:
; HSA: kernarg_segment_byte_size = 40
; OS-MESA3D: kernarg_segment_byte_size = 16
; CO-V2: kernarg_segment_alignment = 4
define amdgpu_kernel void @opencl_test_implicit_alignment_no_explicit_kernargs_round_up() #3 {
  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %val = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

declare ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #0
declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind "amdgpu-implicitarg-num-bytes"="48" }
attributes #3 = { nounwind "amdgpu-implicitarg-num-bytes"="38" }
