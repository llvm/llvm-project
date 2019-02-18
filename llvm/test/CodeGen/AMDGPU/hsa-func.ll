; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=kaveri | FileCheck --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=kaveri | FileCheck --check-prefix=HSA-CI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=carrizo  | FileCheck --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=carrizo | FileCheck --check-prefix=HSA-VI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=kaveri -filetype=obj | llvm-readobj -symbols -s -sd | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=kaveri | llvm-mc -filetype=obj -triple amdgcn--amdhsa -mattr=-code-object-v3 -mcpu=kaveri | llvm-readobj -symbols -s -sd | FileCheck %s --check-prefix=ELF

; The SHT_NOTE section contains the output from the .hsa_code_object_*
; directives.

; ELF: Section {
; ELF: Name: .text
; ELF: Type: SHT_PROGBITS (0x1)
; ELF: Flags [ (0x6)
; ELF: SHF_ALLOC (0x2)
; ELF: SHF_EXECINSTR (0x4)
; ELF: AddressAlignment: 4
; ELF: }

; ELF: SHT_NOTE
; ELF: 0000: 04000000 08000000 01000000 414D4400
; ELF: 0010: 02000000 01000000 04000000 1B000000

; ELF: 0020: 03000000 414D4400 04000700 07000000
; ELF: 0030: 00000000 00000000 414D4400 414D4447
; ELF: 0040: 50550000

; ELF: Symbol {
; ELF: Name: simple
; ELF: Size: 36
; ELF: Type: Function (0x2)
; ELF: }

; HSA: .text
; HSA: .hsa_code_object_version 2,1
; HSA-CI: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
; HSA-VI: .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"

; HSA-NOT: .amdgpu_hsa_kernel simple
; HSA: .globl simple
; HSA: .p2align 2
; HSA: {{^}}simple:
; HSA-NOT: amd_kernel_code_t
; HSA: flat_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[0:1]

; Make sure we are setting the ATC bit:
; Make sure we generate flat store for HSA
; HSA: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}

; HSA: .Lfunc_end0:
; HSA: .size   simple, .Lfunc_end0-simple
; HSA: ; Function info:
; HSA-NOT: COMPUTE_PGM_RSRC2
define void @simple(i32 addrspace(1)* addrspace(4)* %ptr.out) {
entry:
  %out = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %ptr.out
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; Ignore explicit alignment that is too low.
; HSA: .globl simple_align2
; HSA: .p2align 2
define void @simple_align2(i32 addrspace(1)* addrspace(4)* %ptr.out) align 2 {
entry:
  %out = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %ptr.out
  store i32 0, i32 addrspace(1)* %out
  ret void
}
