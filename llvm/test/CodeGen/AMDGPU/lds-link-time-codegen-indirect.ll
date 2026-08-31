; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -amdgpu-enable-lower-module-lds=0 -filetype=obj < %s | llvm-readobj -r --sections - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -amdgpu-enable-lower-module-lds=0 -filetype=asm < %s | FileCheck %s --check-prefix=ASM

; Test that the unified .amdgpu.info section includes address-taken metadata and
; .amdgpu_indirect_call for functions involved in indirect calling.
;
; @callee_vi: void(i32) -- address taken, prototype encoding "vi"
; @caller:    has indirect call to void(i32) -- icall encoding "vi"
; @my_kernel: kernel that passes @callee_vi as a function pointer to @caller

define void @callee_vi(i32 %x) {
  ret void
}

define void @caller(ptr %fptr) {
  call void %fptr(i32 1)
  ret void
}

define amdgpu_kernel void @my_kernel() {
  call void @caller(ptr @callee_vi)
  ret void
}

; CHECK:      Section {
; CHECK:        Name: .amdgpu.info
; CHECK:        Type: SHT_PROGBITS
; CHECK:        Flags [
; CHECK:          SHF_EXCLUDE
; CHECK:        ]

; CHECK-DAG:    R_AMDGPU_ABS64 my_kernel
; CHECK-DAG:    R_AMDGPU_ABS64 caller
; CHECK-DAG:    R_AMDGPU_ABS64 callee_vi

; ASM-DAG:    .amdgpu_info callee_vi
; ASM-DAG:    .amdgpu_flags 0
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:    .amdgpu_typeid "vi"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info caller
; ASM-DAG:    .amdgpu_flags 1
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_indirect_call "vi"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info my_kernel
; ASM-DAG:    .amdgpu_flags 3
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:    .amdgpu_call caller
; ASM-DAG:    .end_amdgpu_info
