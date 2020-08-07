; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-xnack --amdgcn-new-target-id < %s | FileCheck --check-prefixes=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-xnack --amdgcn-new-target-id -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF %s

; ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack-"
; ASM: amdhsa.target: 'amdgcn-amd-amdhsa--gfx900:xnack-'
; ELF: Flags [ (0x62C)
; ELF:   EF_AMDGPU_FEATURE_SRAM_ECC_DEFAULT (0x400)
; ELF:   EF_AMDGPU_FEATURE_XNACK_OFF (0x200)
; ELF:   EF_AMDGPU_MACH_AMDGCN_GFX900 (0x2C)
; ELF: ]

define amdgpu_kernel void @empty() {
entry:
  ret void
}

!0 = !{ i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx900:xnack-" }
!llvm.module.flags = !{ !0 }
