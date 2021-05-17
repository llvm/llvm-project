; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=4 < %s | FileCheck --check-prefixes=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=4 --filetype=obj < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF %s

; ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx900"
; ASM:  amdhsa.target: amdgcn-amd-amdhsa--gfx900
; ASM:  amdhsa.version:
; ASM:    - 1
; ASM:    - 1

; ELF:      OS/ABI: AMDGPU_HSA (0x40)
; ELF:      ABIVersion: 2
; ELF:      Flags [ (0x12C)
; ELF-NEXT:   EF_AMDGPU_FEATURE_XNACK_ANY_V4 (0x100)
; ELF-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX900   (0x2C)
; ELF-NEXT: ]

define void @func0() {
entry:
  ret void
}
