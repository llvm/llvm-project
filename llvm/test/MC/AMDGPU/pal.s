// RUN: llvm-mc -triple amdgcn--amdpal -mcpu=kaveri -show-encoding %s | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdpal -mcpu=kaveri -show-encoding %s | llvm-readobj --symbols -S --sd - | FileCheck %s --check-prefix=ELF

.amd_amdgpu_pal_metadata 0x12345678, 0xfedcba98, 0x2468ace0, 0xfdb97531
// ASM: .amd_amdgpu_pal_metadata 0x12345678,0xfedcba98,0x2468ace0,0xfdb97531
// ELF: SHT_NOTE
// ELF: 0000: 04000000 10000000 0C000000 414D4400
// ELF: 0010: 78563412 98BADCFE E0AC6824 3175B9FD

