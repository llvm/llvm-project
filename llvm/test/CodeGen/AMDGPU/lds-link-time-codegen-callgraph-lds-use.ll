; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r --sections - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=asm < %s | FileCheck %s --check-prefix=ASM

; Test that the unified .amdgpu.info section includes .amdgpu_use for LDS
; alongside resource nodes and call edges. The kernel directly uses an external-linkage
; LDS variable (global-scope), and calls a helper that calls an external.
;
; Expected relationships:
;   - Resources: helper and my_kernel (defined functions with resource info)
;   - Calls: my_kernel -> helper, helper -> extern_func
;   - LDS use: my_kernel -> lds_var

@lds_var = addrspace(3) global [32 x float] poison, align 4

declare void @extern_func()

; CHECK:      Section {
; CHECK:        Name: .amdgpu.info
; CHECK:        Type: SHT_PROGBITS
; CHECK:        Flags [
; CHECK:          SHF_EXCLUDE
; CHECK:        ]

; CHECK-DAG:    R_AMDGPU_ABS64 my_kernel
; CHECK-DAG:    R_AMDGPU_ABS64 helper
; CHECK-DAG:    R_AMDGPU_ABS64 extern_func
; CHECK-DAG:    R_AMDGPU_ABS64 lds_var

; Assembly: per-function .amdgpu_info blocks (target flags derived from e_flags).
; ASM-DAG:    .amdgpu_info helper
; ASM-DAG:    .amdgpu_flags 1
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:    .amdgpu_call extern_func
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info my_kernel
; ASM-DAG:    .amdgpu_flags 3
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:    .amdgpu_use lds_var
; ASM-DAG:    .amdgpu_call helper
; ASM-DAG:    .end_amdgpu_info

define void @helper() {
  call void @extern_func()
  ret void
}

define amdgpu_kernel void @my_kernel() {
  %gep = getelementptr [32 x float], ptr addrspace(3) @lds_var, i32 0, i32 0
  store float 1.0, ptr addrspace(3) %gep
  call void @helper()
  ret void
}
