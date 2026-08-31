; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-lower-module-lds=0 -amdgpu-enable-object-linking < %s | FileCheck -check-prefixes=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-lower-module-lds=0 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r --syms - | FileCheck -check-prefixes=ELF %s

; Uses an external LDS declaration with object linking enabled. The LDS access
; should get a relocation, and the SHN_AMDGPU_LDS symbol should be emitted.
; However, the kernel descriptor should NOT use a symbolic __amdgpu_lds_size
; reference.

@__amdgpu_lds.func = external addrspace(3) global [256 x i8], align 16

; LDS access still generates a relocation.
; ASM-LABEL: {{^}}test_kernel:
; ASM: __amdgpu_lds.func@abs32@lo

; Kernel descriptor uses a constant, not a symbolic reference.
; ASM-NOT: .amdhsa_group_segment_fixed_size __amdgpu_lds_size

; SHN_AMDGPU_LDS symbol is still emitted for the external decl.
; ASM: .amdgpu_lds __amdgpu_lds.func, 256, 16

; --- ELF checks (relocations before symbols in readobj output) ---
; LDS relocation in .text.
; ELF:      R_AMDGPU_ABS32_LO __amdgpu_lds.func

; No symbolic KD reference.
; ELF-NOT:  __amdgpu_lds_size

; SHN_AMDGPU_LDS symbol is present.
; ELF:      Name: __amdgpu_lds.func
; ELF:      Section: Processor Specific (0xFF00)

define amdgpu_kernel void @test_kernel(i32 %idx) {
  %gep = getelementptr [256 x i8], ptr addrspace(3) @__amdgpu_lds.func, i32 0, i32 %idx
  store i8 42, ptr addrspace(3) %gep
  ret void
}
