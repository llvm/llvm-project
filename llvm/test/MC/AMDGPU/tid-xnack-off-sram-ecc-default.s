// RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx900 --mattr=-xnack --amdgcn-new-target-id %s | FileCheck --check-prefixes=ASM %s
// RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx900 --mattr=-xnack --amdgcn-new-target-id -filetype=obj %s | llvm-readobj --file-headers - |FileCheck --check-prefixes=ELF %s

// ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack-"
// ELF: Flags [ (0x62C)
// ELF:   EF_AMDGPU_FEATURE_SRAM_ECC_DEFAULT (0x400)
// ELF:   EF_AMDGPU_FEATURE_XNACK_OFF (0x200)
// ELF:   EF_AMDGPU_MACH_AMDGCN_GFX900 (0x2C)
// ELF: ]

.amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack-"
.text
