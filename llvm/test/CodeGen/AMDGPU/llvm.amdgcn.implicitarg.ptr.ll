; RUN: llc -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,HSA %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MESA %s

; GCN-LABEL: {{^}}kernel_implicitarg_ptr_empty:
; HSA: enable_sgpr_kernarg_segment_ptr = 1
; HSA: kernarg_segment_byte_size = 56
; HSA: kernarg_segment_alignment = 4

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; HSA: s_load_dword s0, s[4:5], 0x0
define amdgpu_kernel void @kernel_implicitarg_ptr_empty() #0 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load = load volatile i32, i32 addrspace(4)* %cast
  ret void
}

; GCN-LABEL: {{^}}kernel_implicitarg_ptr_empty_0implicit:
; HSA: enable_sgpr_kernarg_segment_ptr = 0
; HSA: kernarg_segment_byte_size = 0
; HSA: kernarg_segment_alignment = 4

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; HSA: s_mov_b64 [[NULL:s\[[0-9]+:[0-9]+\]]], 0{{$}}
; HSA: s_load_dword s0, [[NULL]], 0x0

; MESA: s_load_dword s0, s[4:5], 0x0
define amdgpu_kernel void @kernel_implicitarg_ptr_empty_0implicit() #3 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load = load volatile i32, i32 addrspace(4)* %cast
  ret void
}

; GCN-LABEL: {{^}}opencl_kernel_implicitarg_ptr_empty:
; GCN: enable_sgpr_kernarg_segment_ptr = 1

; HSA: kernarg_segment_byte_size = 48
; HSA: kernarg_segment_alignment = 4

; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; HSA: s_load_dword s0, s[4:5], 0x0
define amdgpu_kernel void @opencl_kernel_implicitarg_ptr_empty() #1 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load = load volatile i32, i32 addrspace(4)* %cast
  ret void
}

; GCN-LABEL: {{^}}kernel_implicitarg_ptr:
; GCN: enable_sgpr_kernarg_segment_ptr = 1

; HSA: kernarg_segment_byte_size = 168
; HSA: kernarg_segment_alignment = 4

; MESA: kernarg_segment_byte_size = 128
; MESA: kernarg_segment_alignment = 4

; HSA: s_load_dword s0, s[4:5], 0x1c
define amdgpu_kernel void @kernel_implicitarg_ptr([112 x i8]) #0 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load = load volatile i32, i32 addrspace(4)* %cast
  ret void
}

; GCN-LABEL: {{^}}opencl_kernel_implicitarg_ptr:
; GCN: enable_sgpr_kernarg_segment_ptr = 1

; HSA: kernarg_segment_byte_size = 160
; HSA: kernarg_segment_alignment = 4

; MESA: kernarg_segment_byte_size = 128
; MESA: kernarg_segment_alignment = 4

; HSA: s_load_dword s0, s[4:5], 0x1c
define amdgpu_kernel void @opencl_kernel_implicitarg_ptr([112 x i8]) #1 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load = load volatile i32, i32 addrspace(4)* %cast
  ret void
}

; GCN-LABEL: {{^}}func_implicitarg_ptr:
; GCN: s_waitcnt
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @func_implicitarg_ptr() #0 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load = load volatile i32, i32 addrspace(4)* %cast
  ret void
}

; GCN-LABEL: {{^}}opencl_func_implicitarg_ptr:
; GCN: s_waitcnt
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @opencl_func_implicitarg_ptr() #0 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load = load volatile i32, i32 addrspace(4)* %cast
  ret void
}

; GCN-LABEL: {{^}}kernel_call_implicitarg_ptr_func_empty:
; HSA: enable_sgpr_kernarg_segment_ptr = 1
; HSA: kernarg_segment_byte_size = 56
; HSA: kernarg_segment_alignment = 4

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; GCN: s_mov_b64 s[8:9], s[4:5]
; GCN: s_swappc_b64
define amdgpu_kernel void @kernel_call_implicitarg_ptr_func_empty() #0 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}kernel_call_implicitarg_ptr_func_empty_implicit0:
; HSA: enable_sgpr_kernarg_segment_ptr = 0
; HSA: kernarg_segment_byte_size = 0
; HSA: kernarg_segment_alignment = 4

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; HSA: s_mov_b64 s[8:9], 0{{$}}
; MESA: s_mov_b64 s[8:9], s[4:5]{{$}}
; GCN: s_swappc_b64
define amdgpu_kernel void @kernel_call_implicitarg_ptr_func_empty_implicit0() #3 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}opencl_kernel_call_implicitarg_ptr_func_empty:
; GCN: enable_sgpr_kernarg_segment_ptr = 1
; HSA: kernarg_segment_byte_size = 48
; HSA: kernarg_segment_alignment = 4
; MESA: kernarg_segment_byte_size = 16
; GCN: s_mov_b64 s[8:9], s[4:5]
; GCN-NOT: s4
; GCN-NOT: s5
; GCN: s_swappc_b64
define amdgpu_kernel void @opencl_kernel_call_implicitarg_ptr_func_empty() #1 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}kernel_call_implicitarg_ptr_func:
; GCN: enable_sgpr_kernarg_segment_ptr = 1
; HSA: kernarg_segment_byte_size = 168
; HSA: kernarg_segment_alignment = 4

; MESA: kernarg_segment_byte_size = 128
; MESA: kernarg_segment_alignment = 4

; HSA: s_add_u32 s8, s4, 0x70
; MESA: s_add_u32 s8, s4, 0x70

; GCN: s_addc_u32 s9, s5, 0{{$}}
; GCN: s_swappc_b64
define amdgpu_kernel void @kernel_call_implicitarg_ptr_func([112 x i8]) #0 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}opencl_kernel_call_implicitarg_ptr_func:
; GCN: enable_sgpr_kernarg_segment_ptr = 1
; HSA: kernarg_segment_byte_size = 160
; HSA: kernarg_segment_alignment = 4
; MESA: kernarg_segment_byte_size = 128
; MESA: kernarg_segment_alignment = 4

; GCN: s_add_u32 s8, s4, 0x70
; GCN: s_addc_u32 s9, s5, 0{{$}}
; GCN: s_swappc_b64
define amdgpu_kernel void @opencl_kernel_call_implicitarg_ptr_func([112 x i8]) #1 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}func_call_implicitarg_ptr_func:
; GCN-NOT: s8
; GCN-NOT: s9
; GCN-NOT: s[8:9]
; GCN: s_swappc_b64
; GCN: s_setpc_b64 s[30:31]
define void @func_call_implicitarg_ptr_func() #0 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}opencl_func_call_implicitarg_ptr_func:
; GCN-NOT: s8
; GCN-NOT: s9
; GCN-NOT: s[8:9]
; GCN: s_swappc_b64
; GCN: s_setpc_b64 s[30:31]
define void @opencl_func_call_implicitarg_ptr_func() #0 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}func_kernarg_implicitarg_ptr:
; GCN: s_waitcnt
; GCN-DAG: s_mov_b64 [[NULL:s\[[0-9]+:[0-9]+\]]], 0
; GCN-DAG: s_load_dword s{{[0-9]+}}, [[NULL]], 0x0
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; GCN: s_waitcnt lgkmcnt(0)
define void @func_kernarg_implicitarg_ptr() #0 {
  %kernarg.segment.ptr = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast.kernarg.segment.ptr = bitcast i8 addrspace(4)* %kernarg.segment.ptr to i32 addrspace(4)*
  %cast.implicitarg = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load0 = load volatile i32, i32 addrspace(4)* %cast.kernarg.segment.ptr
  %load1 = load volatile i32, i32 addrspace(4)* %cast.implicitarg
  ret void
}

; GCN-LABEL: {{^}}opencl_func_kernarg_implicitarg_ptr:
; GCN: s_waitcnt
; GCN-DAG: s_mov_b64 [[NULL:s\[[0-9]+:[0-9]+\]]], 0
; GCN-DAG: s_load_dword s{{[0-9]+}}, [[NULL]], 0x0
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; GCN: s_waitcnt lgkmcnt(0)
define void @opencl_func_kernarg_implicitarg_ptr() #0 {
  %kernarg.segment.ptr = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast.kernarg.segment.ptr = bitcast i8 addrspace(4)* %kernarg.segment.ptr to i32 addrspace(4)*
  %cast.implicitarg = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load0 = load volatile i32, i32 addrspace(4)* %cast.kernarg.segment.ptr
  %load1 = load volatile i32, i32 addrspace(4)* %cast.implicitarg
  ret void
}

; GCN-LABEL: {{^}}kernel_call_kernarg_implicitarg_ptr_func:
; GCN: s_add_u32 s8, s4, 0x70
; GCN: s_addc_u32 s9, s5, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kernel_call_kernarg_implicitarg_ptr_func([112 x i8]) #0 {
  call void @func_kernarg_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}kernel_implicitarg_no_struct_align_padding:
; HSA: kernarg_segment_byte_size = 120
; MESA: kernarg_segment_byte_size = 84
; GCN: kernarg_segment_alignment = 6
define amdgpu_kernel void @kernel_implicitarg_no_struct_align_padding(<16 x i32>, i32) #1 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %cast = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %load = load volatile i32, i32 addrspace(4)* %cast
  ret void
}


; HSA-LABEL: Kernels:
; HSA-LABEL: - Name:            kernel_implicitarg_ptr_empty
; HSA: CodeProps:
; HSA: KernargSegmentSize: 56
; HSA: KernargSegmentAlign: 8

; HSA-LABEL: - Name:            kernel_implicitarg_ptr_empty_0implicit
; HSA: KernargSegmentSize: 0
; HSA: KernargSegmentAlign: 4

; HSA-LABEL: - Name:            opencl_kernel_implicitarg_ptr_empty
; HSA: KernargSegmentSize: 48
; HSA: KernargSegmentAlign: 8

; HSA-LABEL: - Name:            kernel_implicitarg_ptr
; HSA: KernargSegmentSize: 168
; HSA: KernargSegmentAlign: 8

; HSA-LABEL: - Name:            opencl_kernel_implicitarg_ptr
; HSA: KernargSegmentSize: 160
; HSA: KernargSegmentAlign: 8

; HSA-LABEL: - Name:            kernel_call_implicitarg_ptr_func_empty
; HSA: KernargSegmentSize: 56
; HSA: KernargSegmentAlign: 8

; HSA-LABEL: - Name:            kernel_call_implicitarg_ptr_func_empty_implicit0
; HSA: KernargSegmentSize: 0
; HSA: KernargSegmentAlign: 4

; HSA-LABEL:  - Name:            opencl_kernel_call_implicitarg_ptr_func_empty
; HSA: KernargSegmentSize: 48
; HSA: KernargSegmentAlign: 8

; HSA-LABEL:  - Name:            kernel_call_implicitarg_ptr_func
; HSA: KernargSegmentSize: 168
; HSA: KernargSegmentAlign: 8

; HSA-LABEL:  - Name:            opencl_kernel_call_implicitarg_ptr_func
; HSA: KernargSegmentSize: 160
; HSA: KernargSegmentAlign: 8

; HSA-LABEL: - Name:            kernel_call_kernarg_implicitarg_ptr_func
; HSA: KernargSegmentSize: 168
; HSA: KernargSegmentAlign: 8

; HSA-LABEL: - Name:            kernel_implicitarg_no_struct_align_padding
; HSA: KernargSegmentSize: 120
; HSA: KernargSegmentAlign: 64

declare i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #2
declare i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #2

attributes #0 = { nounwind noinline }
attributes #1 = { nounwind noinline "amdgpu-implicitarg-num-bytes"="48" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind noinline "amdgpu-implicitarg-num-bytes"="0" }
