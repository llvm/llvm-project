; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V2,HSA,ALL,HSA-NOENV %s
; RUN: llc -mtriple=amdgcn--amdhsa-opencl -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V2,HSA,ALL,HSA-OPENCL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V2,OS-MESA3D,MESA,ALL %s
; RUN: llc -mtriple=amdgcn-mesa-unknown -verify-machineinstrs < %s | FileCheck -check-prefixes=OS-UNKNOWN,MESA,ALL %s

; ALL-LABEL: {{^}}test:
; CO-V2: enable_sgpr_kernarg_segment_ptr = 1
; CO-V2: s_load_dword s{{[0-9]+}}, s[4:5], 0xa

; OS-UNKNOWN: s_load_dword s{{[0-9]+}}, s[0:1], 0xa
define amdgpu_kernel void @test(i32 addrspace(1)* %out) #1 {
  %kernarg.segment.ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr()
  %header.ptr = bitcast i8 addrspace(2)* %kernarg.segment.ptr to i32 addrspace(2)*
  %gep = getelementptr i32, i32 addrspace(2)* %header.ptr, i64 10
  %value = load i32, i32 addrspace(2)* %gep
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit:
; 10 + 9 (36 prepended implicit bytes) + 2(out pointer) = 21 = 0x15
; OS-UNKNOWN: s_load_dword s{{[0-9]+}}, s[0:1], 0x15
define amdgpu_kernel void @test_implicit(i32 addrspace(1)* %out) #1 {
  %implicitarg.ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr()
  %header.ptr = bitcast i8 addrspace(2)* %implicitarg.ptr to i32 addrspace(2)*
  %gep = getelementptr i32, i32 addrspace(2)* %header.ptr, i64 10
  %value = load i32, i32 addrspace(2)* %gep
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit_alignment
; HSA-NOENV: kernarg_segment_byte_size = 10
; HSA-OPENCL: kernarg_segment_byte_size = 48
; OS-MESA3D: kernarg_segment_byte_size = 28
; OS-UNKNOWN: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xc
; HSA: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x4
; OS-MESA3D: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x3
; ALL: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[VAL]]
; MESA: buffer_store_dword [[V_VAL]]
; HSA: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[V_VAL]]
define amdgpu_kernel void @test_implicit_alignment(i32 addrspace(1)* %out, <2 x i8> %in) #1 {
  %implicitarg.ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr()
  %arg.ptr = bitcast i8 addrspace(2)* %implicitarg.ptr to i32 addrspace(2)*
  %val = load i32, i32 addrspace(2)* %arg.ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

declare i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr() #0
declare i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
