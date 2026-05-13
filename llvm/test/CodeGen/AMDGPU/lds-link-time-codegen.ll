; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking < %s | FileCheck -check-prefixes=ASM %s --implicit-check-not=.amdgpu_num_agpr
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r --syms --sections - | FileCheck -check-prefixes=ELF %s

; Test that with object linking enabled, external LDS declarations produce
; @abs32@lo relocations, SHN_AMDGPU_LDS symbols, .amdgpu_lds directives,
; and .amdgpu_use edges in the .amdgpu.info section. Covers multiple LDS
; variables with different sizes and alignments (including zero-sized dynamic
; LDS), usage from both kernels and device functions, and
; group_segment_fixed_size = 0 (linker patches via binary patching).

@lds_large = external addrspace(3) global [256 x i8], align 16
@lds_small = external addrspace(3) global [128 x i8], align 4
@lds_dynamic = external addrspace(3) global [0 x i8], align 8

; Instruction-level relocation checks.
; ASM-LABEL: {{^}}device_func:
; ASM: v_add_u32_e32 v{{[0-9]+}}, lds_large@abs32@lo, v{{[0-9]+}}

; ASM-LABEL: {{^}}test_kernel:
; ASM-DAG: s_add_i32 s{{[0-9]+}}, s{{[0-9]+}}, lds_small@abs32@lo
; ASM-DAG: s_add_i32 s{{[0-9]+}}, s{{[0-9]+}}, lds_dynamic@abs32@lo

; .amdgpu.info section with LDS use edges.
; ASM-DAG: .amdgpu_info device_func
; ASM-DAG:   .amdgpu_flags {{[0-9]+}}
; ASM-DAG:   .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:   .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:   .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:   .amdgpu_use lds_large
; ASM-DAG: .end_amdgpu_info
; ASM-DAG: .amdgpu_info test_kernel
; ASM-DAG:   .amdgpu_flags {{[0-9]+}}
; ASM-DAG:   .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:   .amdgpu_num_sgpr {{[0-9]+}}
; ASM-DAG:   .amdgpu_private_segment_size {{[0-9]+}}
; ASM-DAG:   .amdgpu_use lds_dynamic
; ASM-DAG:   .amdgpu_use lds_small
; ASM-DAG:   .amdgpu_call device_func
; ASM-DAG: .end_amdgpu_info

; SHN_AMDGPU_LDS directives.
; ASM-DAG: .amdgpu_lds lds_large, 256, 16
; ASM-DAG: .amdgpu_lds lds_small, 128, 4
; ASM-DAG: .amdgpu_lds lds_dynamic, 0, 8

; ASM: .group_segment_fixed_size: 0

; .amdgpu.info section exists.
; ELF:      Section {
; ELF:        Name: .amdgpu.info
; ELF:        Type: SHT_PROGBITS
; ELF:        Flags [
; ELF:          SHF_EXCLUDE

; Relocations.
; ELF-DAG: R_AMDGPU_ABS32_LO lds_large
; ELF-DAG: R_AMDGPU_ABS32_LO lds_small
; ELF-DAG: R_AMDGPU_ABS32_LO lds_dynamic
; ELF-DAG: R_AMDGPU_ABS64 device_func
; ELF-DAG: R_AMDGPU_ABS64 test_kernel
; ELF-DAG: R_AMDGPU_ABS64 lds_large
; ELF-DAG: R_AMDGPU_ABS64 lds_small
; ELF-DAG: R_AMDGPU_ABS64 lds_dynamic

; SHN_AMDGPU_LDS symbols.
; ELF-DAG: Name: lds_large
; ELF-DAG: Name: lds_small
; ELF-DAG: Name: lds_dynamic

define void @device_func(i32 %idx) {
  %gep = getelementptr [256 x i8], ptr addrspace(3) @lds_large, i32 0, i32 %idx
  store i8 1, ptr addrspace(3) %gep
  ret void
}

define amdgpu_kernel void @test_kernel(i32 %idx) {
  %gep1 = getelementptr [128 x i8], ptr addrspace(3) @lds_small, i32 0, i32 %idx
  store i8 2, ptr addrspace(3) %gep1
  %gep2 = getelementptr [0 x i8], ptr addrspace(3) @lds_dynamic, i32 0, i32 %idx
  store i8 3, ptr addrspace(3) %gep2
  call void @device_func(i32 %idx)
  ret void
}
