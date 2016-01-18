; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA-CI --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo | FileCheck --check-prefix=HSA-VI --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -filetype=obj  | llvm-readobj -s -sd | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri | llvm-readobj -s -sd | FileCheck %s --check-prefix=ELF

; The SHT_NOTE section contains the output from the .hsa_code_object_*
; directives.

; ELF: Section {
; ELF: Name: .hsatext
; ELF: Type: SHT_PROGBITS (0x1)
; ELF: Flags [ (0xC00007)
; ELF: SHF_ALLOC (0x2)
; ELF: SHF_AMDGPU_HSA_AGENT (0x800000)
; ELF: SHF_AMDGPU_HSA_CODE (0x400000)
; ELF: SHF_EXECINSTR (0x4)
; ELF: SHF_WRITE (0x1)
; ELF: }

; ELF: SHT_NOTE
; ELF: 0000: 04000000 08000000 01000000 414D4400
; ELF: 0010: 01000000 00000000 04000000 1B000000
; ELF: 0020: 03000000 414D4400 04000700 07000000
; ELF: 0030: 00000000 00000000 414D4400 414D4447
; ELF: 0040: 50550000

; HSA: .hsa_code_object_version 1,0
; HSA-CI: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
; HSA-VI: .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"

; HSA: .hsatext

; HSA: {{^}}simple:
; HSA: .amd_kernel_code_t
; HSA: .end_amd_kernel_code_t
; HSA: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[0:1], 0x0

; Make sure we are setting the ATC bit:
; HSA-CI: s_mov_b32 s[[HI:[0-9]]], 0x100f000
; On VI+ we also need to set MTYPE = 2
; HSA-VI: s_mov_b32 s[[HI:[0-9]]], 0x1100f000
; HSA: buffer_store_dword v{{[0-9]+}}, s[0:[[HI]]], 0

define void @simple(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}
