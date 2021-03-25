; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefix=HSA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=4 < %s | FileCheck --check-prefix=HSA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 < %s | FileCheck --check-prefix=HSA %s

declare void @llvm.trap() #0
declare void @llvm.debugtrap() #1

; HSA:      .amdhsa_kernel trap
; HSA-NEXT:     .amdhsa_group_segment_fixed_size 0
; HSA-NEXT:     .amdhsa_private_segment_fixed_size 0
; HSA-NEXT:     .amdhsa_kernarg_size 8
; HSA-NEXT:     .amdhsa_user_sgpr_private_segment_buffer 1
; HSA:      .end_amdhsa_kernel

define amdgpu_kernel void @trap(i32 addrspace(1)* nocapture readonly %arg0) {
  store volatile i32 1, i32 addrspace(1)* %arg0
  call void @llvm.trap()
  unreachable
  store volatile i32 2, i32 addrspace(1)* %arg0
  ret void
}
