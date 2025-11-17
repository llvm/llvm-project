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
