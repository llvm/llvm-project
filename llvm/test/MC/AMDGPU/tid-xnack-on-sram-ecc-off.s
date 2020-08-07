// RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx900 --mattr=+xnack,-sram-ecc --amdgcn-new-target-id %s | FileCheck --check-prefixes=ASM %s
// RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx900 --mattr=+xnack,-sram-ecc --amdgcn-new-target-id -filetype=obj %s | llvm-readobj --file-headers - |FileCheck --check-prefixes=ELF %s

// ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack+:sram-ecc-"
// ELF: Flags [ (0xB2C)
// ELF:   EF_AMDGPU_FEATURE_SRAM_ECC_OFF (0x800)
// ELF:   EF_AMDGPU_FEATURE_XNACK_ON (0x300)
// ELF:   EF_AMDGPU_MACH_AMDGCN_GFX900 (0x2C)
// ELF: ]

.amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack+:sram-ecc-"
.text
