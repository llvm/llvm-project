; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -amdgpu-enable-lower-module-lds=0 -filetype=obj < %s | llvm-readobj -r - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -amdgpu-enable-lower-module-lds=0 -filetype=asm < %s | FileCheck %s --check-prefix=ASM

; Test that address-taken on a declaration (cross-TU scenario) emits a
; .amdgpu_typeid directive for the external function. The function
; is defined in another TU but its address is taken here.

declare void @external_func(i32)

define void @taker(ptr %p) {
  store ptr @external_func, ptr %p
  ret void
}

define amdgpu_kernel void @kern() {
  %p = alloca ptr, addrspace(5)
  call void @taker(ptr addrspace(5) %p)
  ret void
}

; CHECK-DAG: R_AMDGPU_ABS64 external_func
; CHECK-DAG: R_AMDGPU_ABS64 kern
; CHECK-DAG: R_AMDGPU_ABS64 taker

; Assembly: per-function .amdgpu_info blocks (target flags derived from e_flags).
; ASM-DAG:    .amdgpu_info taker
; ASM-DAG:    .amdgpu_flags 0
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:    .amdgpu_typeid "vl"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info kern
; ASM-DAG:    .amdgpu_flags 2
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:    .amdgpu_call taker
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info external_func
; ASM-DAG:    .amdgpu_typeid "vi"
; ASM-DAG:    .end_amdgpu_info
