; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -verify-machineinstrs | FileCheck -check-prefixes=GCN,HSA,COV5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -verify-machineinstrs | FileCheck -check-prefixes=GCN,HSA,COV5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -verify-machineinstrs | FileCheck -check-prefixes=GCN,HSA,COV4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -verify-machineinstrs | FileCheck -check-prefixes=GCN,MESA %s

; GCN-LABEL: {{^}}kernel_implicitarg_ptr_empty:

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; HSA: s_load_dword s0, s[4:5], 0x0

; COV4: .amdhsa_kernarg_size 56
; COV5: .amdhsa_kernarg_size 256
define amdgpu_kernel void @kernel_implicitarg_ptr_empty() #0 {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; GCN-LABEL: {{^}}kernel_implicitarg_ptr_empty_0implicit:
; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; HSA: s_mov_b64 [[NULL:s\[[0-9]+:[0-9]+\]]], 0{{$}}
; HSA: s_load_dword s0, [[NULL]], 0x0

; MESA: s_load_dword s0, s[4:5], 0x0

; COV4: .amdhsa_kernarg_size 0
; COV5: .amdhsa_kernarg_size 0
define amdgpu_kernel void @kernel_implicitarg_ptr_empty_0implicit() #3 {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; GCN-LABEL: {{^}}opencl_kernel_implicitarg_ptr_empty:

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; HSA: s_load_dword s0, s[4:5], 0x0

; HSA: .amdhsa_kernarg_size 48
define amdgpu_kernel void @opencl_kernel_implicitarg_ptr_empty() #1 {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; GCN-LABEL: {{^}}kernel_implicitarg_ptr:

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 128
; MESA: kernarg_segment_alignment = 4

; HSA: s_load_dword s0, s[4:5], 0x1c

; COV4: .amdhsa_kernarg_size 168
; COV5: .amdhsa_kernarg_size 368
define amdgpu_kernel void @kernel_implicitarg_ptr([112 x i8]) #0 {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; GCN-LABEL: {{^}}opencl_kernel_implicitarg_ptr:

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 128
; MESA: kernarg_segment_alignment = 4

; HSA: s_load_dword s0, s[4:5], 0x1c

; HSA: .amdhsa_kernarg_size 160
define amdgpu_kernel void @opencl_kernel_implicitarg_ptr([112 x i8]) #1 {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; GCN-LABEL: {{^}}func_implicitarg_ptr:
; GCN: s_waitcnt
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @func_implicitarg_ptr() #0 {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; GCN-LABEL: {{^}}opencl_func_implicitarg_ptr:
; GCN: s_waitcnt
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @opencl_func_implicitarg_ptr() #0 {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; GCN-LABEL: {{^}}kernel_call_implicitarg_ptr_func_empty:

; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; GCN: s_mov_b64 s[8:9], s[4:5]
; GCN: s_swappc_b64

; COV4: .amdhsa_kernarg_size 56
; COV5: .amdhsa_kernarg_size 256
define amdgpu_kernel void @kernel_call_implicitarg_ptr_func_empty() #0 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}kernel_call_implicitarg_ptr_func_empty_implicit0:
; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; MESA: kernarg_segment_alignment = 4

; HSA: s_mov_b64 s[8:9], 0{{$}}
; MESA: s_mov_b64 s[8:9], s[4:5]{{$}}
; GCN: s_swappc_b64

; HSA: .amdhsa_kernarg_size 0
define amdgpu_kernel void @kernel_call_implicitarg_ptr_func_empty_implicit0() #3 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}opencl_kernel_call_implicitarg_ptr_func_empty:
; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 16
; GCN: s_mov_b64 s[8:9], s[4:5]
; GCN-NOT: s4
; GCN-NOT: s5
; GCN: s_swappc_b64

; HSA: .amdhsa_kernarg_size 48
define amdgpu_kernel void @opencl_kernel_call_implicitarg_ptr_func_empty() #1 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}kernel_call_implicitarg_ptr_func:
; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 128
; MESA: kernarg_segment_alignment = 4

; HSA: s_add_u32 s8, s4, 0x70
; MESA: s_add_u32 s8, s4, 0x70

; GCN: s_addc_u32 s9, s5, 0{{$}}
; GCN: s_swappc_b64

; COV4: .amdhsa_kernarg_size 168
; COV5: .amdhsa_kernarg_size 368
define amdgpu_kernel void @kernel_call_implicitarg_ptr_func([112 x i8]) #0 {
  call void @func_implicitarg_ptr()
  ret void
}

; GCN-LABEL: {{^}}opencl_kernel_call_implicitarg_ptr_func:
; MESA: enable_sgpr_kernarg_segment_ptr = 1
; MESA: kernarg_segment_byte_size = 128
; MESA: kernarg_segment_alignment = 4

; GCN: s_add_u32 s8, s4, 0x70
; GCN: s_addc_u32 s9, s5, 0{{$}}
; GCN: s_swappc_b64

; HSA: .amdhsa_kernarg_size 160
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
  %kernarg.segment.ptr = call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load0 = load volatile i32, ptr addrspace(4) %kernarg.segment.ptr
  %load1 = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; GCN-LABEL: {{^}}opencl_func_kernarg_implicitarg_ptr:
; GCN: s_waitcnt
; GCN-DAG: s_mov_b64 [[NULL:s\[[0-9]+:[0-9]+\]]], 0
; GCN-DAG: s_load_dword s{{[0-9]+}}, [[NULL]], 0x0
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; GCN: s_waitcnt lgkmcnt(0)
define void @opencl_func_kernarg_implicitarg_ptr() #0 {
  %kernarg.segment.ptr = call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load0 = load volatile i32, ptr addrspace(4) %kernarg.segment.ptr
  %load1 = load volatile i32, ptr addrspace(4) %implicitarg.ptr
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
; MESA: kernarg_segment_byte_size = 84
; MESA: kernarg_segment_alignment = 6

; HSA: .amdhsa_kernarg_size 120
define amdgpu_kernel void @kernel_implicitarg_no_struct_align_padding(<16 x i32>, i32) #1 {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load volatile i32, ptr addrspace(4) %implicitarg.ptr
  ret void
}

; HSA-LABEL:   amdhsa.kernels:
; HSA:         .kernarg_segment_align: 8
; COV5-NEXT:    .kernarg_segment_size: 256
; COV4-NEXT:    .kernarg_segment_size: 56
; HSA-LABEL:   .name:           kernel_implicitarg_ptr_empty

; HSA:         .kernarg_segment_align: 4
; HSA-NEXT:    .kernarg_segment_size: 0
; HSA-LABEL:   .name:           kernel_implicitarg_ptr_empty_0implicit

; HSA:         .kernarg_segment_align: 8
; HSA-NEXT:    .kernarg_segment_size: 48
; HSA-LABEL:   .name:           opencl_kernel_implicitarg_ptr_empty

; HSA:         .kernarg_segment_align: 8
; COV5-NEXT:    .kernarg_segment_size: 368
; COV4-NEXT:    .kernarg_segment_size: 168
; HSA-LABEL:   .name:           kernel_implicitarg_ptr

; HSA:         .kernarg_segment_align: 8
; HSA-NEXT:    .kernarg_segment_size: 160
; HSA-LABEL:   .name:           opencl_kernel_implicitarg_ptr

; HSA:         .kernarg_segment_align: 8
; COV5-NEXT:    .kernarg_segment_size: 256
; COV4-NEXT:    .kernarg_segment_size: 56
; HSA-LABEL:   .name:           kernel_call_implicitarg_ptr_func_empty

; HSA:         .kernarg_segment_align: 4
; HSA-NEXT:    .kernarg_segment_size: 0
; HSA-LABEL:   .name:           kernel_call_implicitarg_ptr_func_empty_implicit0

; HSA:         .kernarg_segment_align: 8
; HSA-NEXT:    .kernarg_segment_size: 48
; HSA-LABEL:   .name:           opencl_kernel_call_implicitarg_ptr_func_empty

; HSA:         .kernarg_segment_align: 8
; COV5-NEXT:    .kernarg_segment_size: 368
; COV4-NEXT:    .kernarg_segment_size: 168
; HSA-LABEL:   .name:           kernel_call_implicitarg_ptr_func

; HSA:         .kernarg_segment_align: 8
; HSA-NEXT:    .kernarg_segment_size: 160
; HSA-LABEL:   .name:           opencl_kernel_call_implicitarg_ptr_func

; HSA:         .kernarg_segment_align: 8
; COV5-NEXT:    .kernarg_segment_size: 368
; COV4-NEXT:    .kernarg_segment_size: 168
; HSA-LABEL:   .name:           kernel_call_kernarg_implicitarg_ptr_func

; HSA:         .kernarg_segment_align: 64
; HSA-NEXT:    .kernarg_segment_size: 120
; HSA-LABEL:   .name:           kernel_implicitarg_no_struct_align_padding

declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #2
declare ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #2

attributes #0 = { nounwind noinline }
attributes #1 = { nounwind noinline "amdgpu-implicitarg-num-bytes"="48" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind noinline "amdgpu-implicitarg-num-bytes"="0" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
