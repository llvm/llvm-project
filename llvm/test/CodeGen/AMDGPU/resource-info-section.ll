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
; ASM-NEXT: .has_recursion 0
; ASM-NEXT: .has_indirect_call 0
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
; ASM-NEXT: .has_recursion 0
; ASM-NEXT: .has_indirect_call 0
; ASM-NEXT: .end_amdgpu_resource_usage
define void @use_vcc() {
  call void asm sideeffect "", "~{vcc}" ()
  ret void
}

; ASM-LABEL: {{^}}caller:
; ASM: .amdgpu_resource_usage caller
; ASM:      .callee use_vcc
; ASM-NEXT: .end_amdgpu_resource_usage
define void @caller() {
  call void @use_vcc()
  ret void
}

; ASM-LABEL: {{^}}kernel:
; ASM: .amdgpu_resource_usage kernel
; ASM:      .callee caller
; ASM-NEXT: .end_amdgpu_resource_usage
define amdgpu_kernel void @kernel() {
  call void @caller()
  ret void
}


; ASM-LABEL: {{^}}rcaller2:
; ASM: .amdgpu_resource_usage rcaller2
; ASM:      .callee rcaller1
; ASM-NEXT: .end_amdgpu_resource_usage
; ASM-LABEL: {{^}}rcaller1:
; ASM: .amdgpu_resource_usage rcaller1
; ASM:      .callee rcaller2
; ASM-NEXT: .end_amdgpu_resource_usage
define void @rcaller1() {
  call void @rcaller2()
  ret void
}
define void @rcaller2() {
  call void @rcaller1()
  ret void
}

; ASM-LABEL: {{^}}kernel_recurse:
; ASM: .amdgpu_resource_usage kernel
; ASM:      .callee rcaller1
; ASM-NEXT: .end_amdgpu_resource_usage
define amdgpu_kernel void @kernel_recurse() {
  call void @rcaller1()
  ret void
}

; RELOC:      Relocation section '.rela.AMDGPU.resource_info'
; RELOC:      0000000000000000 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} leaf + 0
; RELOC-NEXT: 0000000000000018 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} use_vcc + 0
; RELOC-NEXT: 0000000000000030 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} caller + 0
; RELOC-NEXT: 0000000000000030 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} use_vcc + 0
; RELOC-NEXT: 0000000000000048 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} kernel + 0
; RELOC-NEXT: 0000000000000048 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} caller + 0
; RELOC-NEXT: 0000000000000060 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} rcaller2 + 0
; RELOC-NEXT: 0000000000000060 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} rcaller1 + 0
; RELOC-NEXT: 0000000000000078 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} rcaller1 + 0
; RELOC-NEXT: 0000000000000078 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} rcaller2 + 0
; RELOC-NEXT: 0000000000000090 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} kernel_recurse + 0
; RELOC-NEXT: 0000000000000090 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} rcaller1 + 0

