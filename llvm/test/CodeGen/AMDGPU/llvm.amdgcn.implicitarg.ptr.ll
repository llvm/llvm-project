; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,HSA,HSA-NOENV %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa-opencl -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,HSA,HSA-OPENCL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MESA %s

; GCN-LABEL: {{^}}kernel_implicitarg_ptr_empty:
; GCN: enable_sgpr_kernarg_segment_ptr = 1

; HSA-NOENV: kernarg_segment_byte_size = 0
; HSA-OPENCL: kernarg_segment_byte_size = 32
; MESA: kernarg_segment_byte_size = 16

; HSA: s_load_dword s0, s[4:5], 0x0
define amdgpu_kernel void @kernel_implicitarg_ptr_empty() #0 {
  %implicitarg.ptr = call i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(2)* %implicitarg.ptr to i32 addrspace(2)*
  %load = load volatile i32, i32 addrspace(2)* %cast
  ret void
}

; GCN-LABEL: {{^}}kernel_implicitarg_ptr:
; GCN: enable_sgpr_kernarg_segment_ptr = 1

; HSA-NOENV: kernarg_segment_byte_size = 112
; HSA-OPENCL: kernarg_segment_byte_size = 144
; MESA: kernarg_segment_byte_size = 464

; HSA: s_load_dword s0, s[4:5], 0x1c
define amdgpu_kernel void @kernel_implicitarg_ptr([112 x i8]) #0 {
  %implicitarg.ptr = call i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(2)* %implicitarg.ptr to i32 addrspace(2)*
  %load = load volatile i32, i32 addrspace(2)* %cast
  ret void
}

declare i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr() #2

attributes #0 = { nounwind noinline }
attributes #1 = { nounwind noinline }
attributes #2 = { nounwind readnone speculatable }
