; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stop-after=amdgpu-isel %s -o - | FileCheck --check-prefix=MIR %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stop-after=amdgpu-isel -o %t.mir %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -start-after=amdgpu-isel -verify-machineinstrs %t.mir -o - | FileCheck --check-prefix=ASM %s

; Test that kernarg preloading information is correctly serialized to MIR and
; can be round-tripped through MIR serialization/deserialization.

; MIR-LABEL: name: kernarg_preload_single_arg
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 1

; ASM-LABEL: kernarg_preload_single_arg:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 1
; ASM: .amdhsa_user_sgpr_kernarg_preload_offset 0
define amdgpu_kernel void @kernarg_preload_single_arg(i32 inreg %arg0) {
entry:
  %val = add i32 %arg0, 1
  store i32 %val, ptr addrspace(1) null
  ret void
}

; MIR-LABEL: name: kernarg_preload_multiple_args_unaligned
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 5

; ASM-LABEL: kernarg_preload_multiple_args_unaligned:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 5
; ASM: .amdhsa_user_sgpr_kernarg_preload_offset 0
define amdgpu_kernel void @kernarg_preload_multiple_args_unaligned(i32 inreg %arg0, i64 inreg %arg1, i32 inreg %arg2) {
entry:
  %val = add i32 %arg0, %arg2
  store i32 %val, ptr addrspace(1) null
  ret void
}

; MIR-LABEL: name: kernarg_preload_multiple_args_aligned
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 4

; ASM-LABEL: kernarg_preload_multiple_args_aligned:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 4
; ASM: .amdhsa_user_sgpr_kernarg_preload_offset 0
define amdgpu_kernel void @kernarg_preload_multiple_args_aligned(i64 inreg %arg0, i32 inreg %arg1, i32 inreg %arg2) {
entry:
  %val = add i32 %arg1, %arg2
  store i32 %val, ptr addrspace(1) null
  ret void
}

; MIR-LABEL: name: kernarg_preload_with_ptr
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 2

; ASM-LABEL: kernarg_preload_with_ptr:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 2
; ASM: .amdhsa_user_sgpr_kernarg_preload_offset 0
define amdgpu_kernel void @kernarg_preload_with_ptr(ptr inreg %ptr) {
entry:
  %val = load i32, ptr %ptr
  %add = add i32 %val, 1
  store i32 %add, ptr %ptr
  ret void
}

; MIR-LABEL: name: kernarg_no_preload
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; MIR-NOT: firstKernArgPreloadReg
; MIR: numKernargPreloadSGPRs: 0

; ASM-LABEL: kernarg_no_preload:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 0
define amdgpu_kernel void @kernarg_no_preload(i32 %arg0) {
entry:
  %val = add i32 %arg0, 1
  store i32 %val, ptr addrspace(1) null
  ret void
}

; MIR-LABEL: name: kernarg_preload_mixed
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 2

; ASM-LABEL: kernarg_preload_mixed:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 2
define amdgpu_kernel void @kernarg_preload_mixed(i32 inreg %arg0, i32 inreg %arg1, i32 %arg2) {
entry:
  %val = add i32 %arg0, %arg1
  %val2 = add i32 %val, %arg2
  store i32 %val2, ptr addrspace(1) null
  ret void
}

; MIR-LABEL: name: kernarg_preload_with_dispatch_ptr
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: dispatchPtr: { reg: '$sgpr0_sgpr1' }
; MIR: kernargSegmentPtr: { reg: '$sgpr2_sgpr3' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr4' }
; MIR: numKernargPreloadSGPRs: 2

; ASM-LABEL: kernarg_preload_with_dispatch_ptr:
; ASM: .amdhsa_user_sgpr_dispatch_ptr 1
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 2

define amdgpu_kernel void @kernarg_preload_with_dispatch_ptr(i64 inreg %arg0) #0 {
entry:
  %val = add i64 %arg0, 1
  store i64 %val, ptr addrspace(1) null
  ret void
}

attributes #0 = { "amdgpu-dispatch-ptr" "amdgpu-no-queue-ptr" "amdgpu-no-dispatch-id" }

; MIR-LABEL: name: kernarg_preload_with_queue_ptr
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: queuePtr: { reg: '$sgpr0_sgpr1' }
; MIR: kernargSegmentPtr: { reg: '$sgpr2_sgpr3' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr4' }
; MIR: numKernargPreloadSGPRs: 1

; ASM-LABEL: kernarg_preload_with_queue_ptr:
; ASM: .amdhsa_user_sgpr_queue_ptr 1
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 1

define amdgpu_kernel void @kernarg_preload_with_queue_ptr(i32 inreg %arg0) #1 {
entry:
  %val = add i32 %arg0, 1
  store i32 %val, ptr addrspace(1) null
  ret void
}

attributes #1 = { "amdgpu-queue-ptr" "amdgpu-no-dispatch-ptr" "amdgpu-no-dispatch-id" }

; MIR-LABEL: name: kernarg_preload_with_multiple_user_sgprs
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: dispatchPtr: { reg: '$sgpr0_sgpr1' }
; MIR: queuePtr: { reg: '$sgpr2_sgpr3' }
; MIR: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; MIR: dispatchID: { reg: '$sgpr6_sgpr7' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 2

; ASM-LABEL: kernarg_preload_with_multiple_user_sgprs:
; ASM: .amdhsa_user_sgpr_dispatch_ptr 1
; ASM: .amdhsa_user_sgpr_queue_ptr 1
; ASM: .amdhsa_user_sgpr_dispatch_id 1
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 2

define amdgpu_kernel void @kernarg_preload_with_multiple_user_sgprs(i64 inreg %arg0) #5 {
entry:
  %val = add i64 %arg0, 1
  store i64 %val, ptr addrspace(1) null
  ret void
}

attributes #2 = { "amdgpu-dispatch-ptr" "amdgpu-queue-ptr" "amdgpu-dispatch-id" }

; MIR-LABEL: name: kernarg_preload_without_user_sgprs
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: kernargSegmentPtr: { reg: '$sgpr0_sgpr1' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr2' }
; MIR: numKernargPreloadSGPRs: 1

; ASM-LABEL: kernarg_preload_without_user_sgprs:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 1

define amdgpu_kernel void @kernarg_preload_without_user_sgprs(i32 inreg %arg0) #3 {
entry:
  %val = add i32 %arg0, 1
  store i32 %val, ptr addrspace(1) null
  ret void
}

attributes #3 = { "amdgpu-no-queue-ptr" "amdgpu-no-dispatch-ptr" "amdgpu-no-dispatch-id" }

; MIR-LABEL: name: kernarg_preload_max_args
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: dispatchPtr: { reg: '$sgpr0_sgpr1' }
; MIR: queuePtr: { reg: '$sgpr2_sgpr3' }
; MIR: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; MIR: dispatchID: { reg: '$sgpr6_sgpr7' }
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 8

; ASM-LABEL: kernarg_preload_max_args:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 8

define amdgpu_kernel void @kernarg_preload_max_args(
    i32 inreg %a0, i32 inreg %a1, i32 inreg %a2, i32 inreg %a3,
    i32 inreg %a4, i32 inreg %a5, i32 inreg %a6, i32 inreg %a7,
    i32 inreg %a8, i32 inreg %a9, i32 inreg %a10, i32 inreg %a11,
    i32 inreg %a12, i32 inreg %a13, i32 inreg %a14, i32 inreg %a15) {
entry:
  ret void
}

; MIR-LABEL: name: kernarg_preload_mixed_inreg_and_stack
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 2

; ASM-LABEL: kernarg_preload_mixed_inreg_and_stack:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 2

define amdgpu_kernel void @kernarg_preload_mixed_inreg_and_stack(
    i32 inreg %preload0,
    i32 inreg %preload1,
    i32 %stack0,
    i32 %stack1) {
entry:
  %val = add i32 %preload0, %preload1
  %val2 = add i32 %val, %stack0
  %val3 = add i32 %val2, %stack1
  store i32 %val3, ptr addrspace(1) null
  ret void
}

; MIR-LABEL: name: kernarg_preload_vector_types
; MIR: machineFunctionInfo:
; MIR: argumentInfo:
; MIR: firstKernArgPreloadReg: { reg: '$sgpr8' }
; MIR: numKernargPreloadSGPRs: 4

; ASM-LABEL: kernarg_preload_vector_types:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 4

define amdgpu_kernel void @kernarg_preload_vector_types(<4 x i32> inreg %vec) {
entry:
  %elem = extractelement <4 x i32> %vec, i32 0
  store i32 %elem, ptr addrspace(1) null
  ret void
}
