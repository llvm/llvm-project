; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=ALL-ASM,OSABI-AMDHSA-ASM %s
; RUN: llc -filetype=obj -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -mattr=+code-object-v3 < %s | llvm-readelf --notes -relocations -sections -symbols | FileCheck --check-prefixes=ALL-ELF,OSABI-AMDHSA-ELF %s

; ALL-ASM-LABEL: {{^}}fadd:

; OSABI-AMDHSA-ASM-NOT: .hsa_code_object_version
; OSABI-AMDHSA-ASM-NOT: .hsa_code_object_isa
; OSABI-AMDHSA-ASM-NOT: .amdgpu_hsa_kernel
; OSABI-AMDHSA-ASM-NOT: .amd_kernel_code_t

; OSABI-AMDHSA-ASM: s_endpgm
; OSABI-AMDHSA-ASM: .section .rodata,#alloc
; OSABI-AMDHSA-ASM: .p2align 6
; OSABI-AMDHSA-ASM: .amdhsa_kernel fadd
; OSABI-AMDHSA-ASM:     .amdhsa_user_sgpr_private_segment_buffer 1
; OSABI-AMDHSA-ASM:     .amdhsa_user_sgpr_kernarg_segment_ptr 1
; OSABI-AMDHSA-ASM:     .amdhsa_next_free_vgpr 3
; OSABI-AMDHSA-ASM:     .amdhsa_next_free_sgpr 8
; OSABI-AMDHSA-ASM:     .amdhsa_reserve_vcc 0
; OSABI-AMDHSA-ASM:     .amdhsa_reserve_flat_scratch 0
; OSABI-AMDHSA-ASM: .end_amdhsa_kernel
; OSABI-AMDHSA-ASM: .text

; ALL-ASM-LABEL: {{^}}fsub:

; OSABI-AMDHSA-ASM-NOT: .amdgpu_hsa_kernel
; OSABI-AMDHSA-ASM-NOT: .amd_kernel_code_t

; OSABI-AMDHSA-ASM: s_endpgm
; OSABI-AMDHSA-ASM: .section .rodata,#alloc
; OSABI-AMDHSA-ASM: .p2align 6
; OSABI-AMDHSA-ASM: .amdhsa_kernel fsub
; OSABI-AMDHSA-ASM:     .amdhsa_user_sgpr_private_segment_buffer 1
; OSABI-AMDHSA-ASM:     .amdhsa_user_sgpr_kernarg_segment_ptr 1
; OSABI-AMDHSA-ASM:     .amdhsa_next_free_vgpr 3
; OSABI-AMDHSA-ASM:     .amdhsa_next_free_sgpr 8
; OSABI-AMDHSA-ASM:     .amdhsa_reserve_vcc 0
; OSABI-AMDHSA-ASM:     .amdhsa_reserve_flat_scratch 0
; OSABI-AMDHSA-ASM: .end_amdhsa_kernel
; OSABI-AMDHSA-ASM: .text

; OSABI-AMDHSA-ASM-NOT: .hsa_code_object_version
; OSABI-AMDHSA-ASM-NOT: .hsa_code_object_isa
; OSABI-AMDHSA-ASM-NOT: .amd_amdgpu_isa
; OSABI-AMDHSA-ASM-NOT: .amd_amdgpu_hsa_metadata
; OSABI-AMDHSA-ASM-NOT: .amd_amdgpu_pal_metadata

; OSABI-AMDHSA-ELF: Section Headers
; OSABI-AMDHSA-ELF: .text   PROGBITS {{[0-9]+}} {{[0-9]+}} {{[0-9a-f]+}} {{[0-9]+}} AX {{[0-9]+}} {{[0-9]+}} 256
; OSABI-AMDHSA-ELF: .rodata PROGBITS {{[0-9]+}} {{[0-9]+}} {{[0-9a-f]+}} {{[0-9]+}}  A {{[0-9]+}} {{[0-9]+}} 64

; OSABI-AMDHSA-ELF: Relocation section '.rela.rodata' at offset
; OSABI-AMDHSA-ELF: 0000000000000010 0000000100000005 R_AMDGPU_REL64 0000000000000000 fadd + 10
; OSABI-AMDHSA-ELF: 0000000000000050 0000000300000005 R_AMDGPU_REL64 0000000000000100 fsub + 10

; OSABI-AMDHSA-ELF: Symbol table '.symtab' contains {{[0-9]+}} entries
; OSABI-AMDHSA-ELF: {{[0-9]+}}: 0000000000000000 {{[0-9]+}} FUNC   GLOBAL PROTECTED {{[0-9]+}} fadd
; OSABI-AMDHSA-ELF: {{[0-9]+}}: 0000000000000000 64         OBJECT GLOBAL DEFAULT   {{[0-9]+}} fadd.kd
; OSABI-AMDHSA-ELF: {{[0-9]+}}: 0000000000000100 {{[0-9]+}} FUNC   GLOBAL PROTECTED {{[0-9]+}} fsub
; OSABI-AMDHSA-ELF: {{[0-9]+}}: 0000000000000040 64         OBJECT GLOBAL DEFAULT   {{[0-9]+}} fsub.kd

; OSABI-AMDHSA-ELF: Displaying notes found at file offset
; OSABI-AMDHSA-ELF: AMDGPU 0x{{[0-9a-f]+}} NT_AMDGPU_METADATA (AMDGPU Metadata)

define amdgpu_kernel void @fadd(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %b) {
entry:
  %a.val = load float, float addrspace(1)* %a
  %b.val = load float, float addrspace(1)* %b
  %r.val = fadd float %a.val, %b.val
  store float %r.val, float addrspace(1)* %r
  ret void
}

define amdgpu_kernel void @fsub(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %b) {
entry:
  %a.val = load float, float addrspace(1)* %a
  %b.val = load float, float addrspace(1)* %b
  %r.val = fsub float %a.val, %b.val
  store float %r.val, float addrspace(1)* %r
  ret void
}
