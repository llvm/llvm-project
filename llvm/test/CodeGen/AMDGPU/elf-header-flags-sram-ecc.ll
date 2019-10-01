; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx902 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-SRAM-ECC-GFX902 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx902 -mattr=-sram-ecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-SRAM-ECC-GFX902 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx902 -mattr=+sram-ecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-GFX902 %s

; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=-sram-ecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=+sram-ecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=+sram-ecc,+xnack < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-XNACK-GFX906 %s

; NO-SRAM-ECC-GFX902:      Flags [
; NO-SRAM-ECC-GFX902-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX902 (0x2D)
; NO-SRAM-ECC-GFX902-NEXT:   EF_AMDGPU_XNACK              (0x100)
; NO-SRAM-ECC-GFX902-NEXT: ]

; SRAM-ECC-GFX902:      Flags [
; SRAM-ECC-GFX902-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX902 (0x2D)
; SRAM-ECC-GFX902-NEXT:   EF_AMDGPU_SRAM_ECC           (0x200)
; SRAM-ECC-GFX902-NEXT:   EF_AMDGPU_XNACK              (0x100)
; SRAM-ECC-GFX902-NEXT: ]

; NO-SRAM-ECC-GFX906:      Flags [
; NO-SRAM-ECC-GFX906-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX906 (0x2F)
; NO-SRAM-ECC-GFX906-NEXT: ]

; SRAM-ECC-GFX906:      Flags [
; SRAM-ECC-GFX906-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX906 (0x2F)
; SRAM-ECC-GFX906-NEXT:   EF_AMDGPU_SRAM_ECC           (0x200)
; SRAM-ECC-GFX906-NEXT: ]

; SRAM-ECC-XNACK-GFX906:      Flags [
; SRAM-ECC-XNACK-GFX906-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX906 (0x2F)
; SRAM-ECC-XNACK-GFX906-NEXT:   EF_AMDGPU_SRAM_ECC           (0x200)
; SRAM-ECC-XNACK-GFX906-NEXT:   EF_AMDGPU_XNACK              (0x100)
; SRAM-ECC-XNACK-GFX906-NEXT: ]

define amdgpu_kernel void @elf_header() {
  ret void
}
