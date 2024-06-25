; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA-CI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo  | FileCheck --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo | FileCheck --check-prefix=HSA-VI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -filetype=obj | llvm-readobj --symbols -S --sd - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | llvm-mc -filetype=obj -triple amdgcn--amdhsa --amdhsa-code-object-version=4 -mcpu=kaveri | llvm-readobj --symbols -S --sd - | FileCheck %s --check-prefix=ELF

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
; ELF: 0000: 07000000 4F000000 20000000 414D4447
; ELF: 0010: 50550000 83AE616D 64687361 2E6B6572
; ELF: 0020: 6E656C73 90AD616D 64687361 2E746172
; ELF: 0030: 676574BD 616D6467 636E2D75 6E6B6E6F
; ELF: 0040: 776E2D61 6D646873 612D2D67 66783730
; ELF: 0050: 30AE616D 64687361 2E766572 73696F6E
; ELF: 0060: 92010100

; ELF: Symbol {
; ELF: Name: simple
; ELF: Size: 36
; ELF: Type: Function (0x2)
; ELF: }

; HSA: .text
; HSA-CI: .amdgcn_target "amdgcn-unknown-amdhsa--gfx700" 
; HSA-VI: .amdgcn_target "amdgcn-unknown-amdhsa--gfx801" 

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
define void @simple(ptr addrspace(4) %ptr.out) {
entry:
  %out = load ptr addrspace(1), ptr addrspace(4) %ptr.out
  store i32 0, ptr addrspace(1) %out
  ret void
}

; Ignore explicit alignment that is too low.
; HSA: .globl simple_align2
; HSA: .p2align 2
define void @simple_align2(ptr addrspace(4) %ptr.out) align 2 {
entry:
  %out = load ptr addrspace(1), ptr addrspace(4) %ptr.out
  store i32 0, ptr addrspace(1) %out
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
