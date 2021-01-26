; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=-sramecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=+sramecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=+sramecc,+xnack < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-XNACK-GFX906 %s

; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx908 -mattr=+sramecc < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=SRAM-ECC-GFX908 %s

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

; SRAM-ECC-GFX908: Flags [ (0x230)
; SRAM-ECC-GFX908:    EF_AMDGPU_MACH_AMDGCN_GFX908 (0x30)
; SRAM-ECC-GFX908:    EF_AMDGPU_SRAM_ECC (0x200)
; SRAM-ECC-GFX908:  ]

define amdgpu_kernel void @elf_header() {
  ret void
}
