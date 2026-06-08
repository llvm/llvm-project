; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r --sections - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=asm < %s | FileCheck %s --check-prefix=ASM --implicit-check-not=.amdgpu_num_agpr

; Test that the unified .amdgpu.info section (.amdgpu_info blocks in assembly) is
; emitted with correct relocations when object linking is enabled.

declare void @extern_func()
declare void @tail_extern()

; The .amdgpu.info section should exist as SHT_PROGBITS with SHF_EXCLUDE.
; CHECK:      Section {
; CHECK:        Name: .amdgpu.info
; CHECK:        Type: SHT_PROGBITS
; CHECK:        Flags [
; CHECK:          SHF_EXCLUDE
; CHECK:        ]

; Symbol references in the binary resource metadata still use R_AMDGPU_ABS64 relocations.
; CHECK-DAG:    R_AMDGPU_ABS64 my_kernel
; CHECK-DAG:    R_AMDGPU_ABS64 helper
; CHECK-DAG:    R_AMDGPU_ABS64 extern_func
; COM: Tail-call callee must still be recorded as an INFO_CALL edge.
; CHECK-DAG:    R_AMDGPU_ABS64 tail_helper
; CHECK-DAG:    R_AMDGPU_ABS64 tail_extern

; COM: Assembly: per-function .amdgpu_info blocks (target flags derived from
; COM: e_flags).
; ASM-DAG:    .amdgpu_info helper
; ASM-DAG:    .amdgpu_flags {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:    .amdgpu_call extern_func
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info my_kernel
; ASM-DAG:    .amdgpu_flags {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:    .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:    .amdgpu_call helper
; ASM-DAG:    .end_amdgpu_info

; COM: A tail call is lowered to SI_TCRETURN (isCall = 1). Verify that the
; COM: callee edge is still captured in the .amdgpu_info block of the caller.
; ASM-DAG:    .amdgpu_info tail_helper
; ASM-DAG:    .amdgpu_call tail_extern
; ASM-DAG:    .end_amdgpu_info

define void @helper() {
  call void @extern_func()
  ret void
}

define amdgpu_kernel void @my_kernel() {
  call void @helper()
  ret void
}

define void @tail_helper() {
  tail call void @tail_extern()
  ret void
}
