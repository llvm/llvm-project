; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-ipra=0 < %s | FileCheck -check-prefix=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-ipra=0 -filetype=obj < %s -o %t
; RUN: llvm-readelf -r %t | FileCheck -check-prefix=RELOC %s

; ASM-LABEL: {{^}}leaf:
; ASM: .amdgpu_resource_usage leaf
; ASM-NEXT: .num_vgpr 0
; ASM-NEXT: .num_agpr 0
; ASM-NEXT: .num_sgpr 32
; ASM-NEXT: .named_barrier 0
; ASM-NEXT: .private_seg_size 0
; ASM-NEXT: .uses_vcc 0
; ASM-NEXT: .uses_flat_scratch 0
; ASM-NEXT: .has_dyn_sized_stack 0
; ASM-NEXT: .end_amdgpu_resource_usage
define void @leaf() {
  ret void
}

; ASM-LABEL: {{^}}use_vcc:
; ASM: .amdgpu_resource_usage use_vcc
; ASM-NEXT: .num_vgpr 0
; ASM-NEXT: .num_agpr 0
; ASM-NEXT: .num_sgpr 32
; ASM-NEXT: .named_barrier 0
; ASM-NEXT: .private_seg_size 0
; ASM-NEXT: .uses_vcc 1
; ASM-NEXT: .uses_flat_scratch 0
; ASM-NEXT: .has_dyn_sized_stack 0
; ASM-NEXT: .end_amdgpu_resource_usage
define void @use_vcc() {
  call void asm sideeffect "", "~{vcc}" ()
  ret void
}

; RELOC:      Relocation section '.rela.AMDGPU.resource_usage'
; RELOC:      0000000000000000 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} leaf + 0
; RELOC-NEXT: 0000000000000018 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} use_vcc + 0

