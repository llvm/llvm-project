// RUN: llvm-mc -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | llvm-readobj -s -sd | FileCheck %s --check-prefix=ELF

// For compatibility reasons we treat convert .text sections to .hsatext

// ELF: Section {

// ELF: Name: .text
// ELF: Type: SHT_PROGBITS (0x1)
// ELF: Flags [ (0x6)
// ELF: SHF_ALLOC (0x2)
// ELF: SHF_EXECINSTR (0x4)
// ELF: Size: 260
// ELF: }

.text
// ASM: .text

.hsa_code_object_version 1,0
// ASM: .hsa_code_object_version 1,0

.hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
// ASM: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"

.amd_kernel_code_t
.end_amd_kernel_code_t

s_endpgm
