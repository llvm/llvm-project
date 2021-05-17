; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=-sramecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=+sramecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 -mattr=+sramecc,+xnack < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-XNACK-GFX906 %s

; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx908 < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=SRAM-ECC-GFX908 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx908 -mattr=+sramecc < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=SRAM-ECC-GFX908 %s

; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx90a < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=SRAM-ECC-GFX90A %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx90a < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=SRAM-ECC-GFX90A %s

; NO-SRAM-ECC-GFX906:      Flags [
; NO-SRAM-ECC-GFX906-NEXT:   EF_AMDGPU_FEATURE_XNACK_V3   (0x100)
; NO-SRAM-ECC-GFX906-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX906 (0x2F)
; NO-SRAM-ECC-GFX906-NEXT: ]

; SRAM-ECC-GFX906:      Flags [
; SRAM-ECC-GFX906-NEXT:   EF_AMDGPU_FEATURE_SRAMECC_V3 (0x200)
; SRAM-ECC-GFX906-NEXT:   EF_AMDGPU_FEATURE_XNACK_V3   (0x100)
; SRAM-ECC-GFX906-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX906 (0x2F)
; SRAM-ECC-GFX906-NEXT: ]

; SRAM-ECC-XNACK-GFX906:      Flags [
; SRAM-ECC-XNACK-GFX906-NEXT:   EF_AMDGPU_FEATURE_SRAMECC_V3 (0x200)
; SRAM-ECC-XNACK-GFX906-NEXT:   EF_AMDGPU_FEATURE_XNACK_V3   (0x100)
; SRAM-ECC-XNACK-GFX906-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX906 (0x2F)
; SRAM-ECC-XNACK-GFX906-NEXT: ]

; SRAM-ECC-GFX908: Flags [
; SRAM-ECC-GFX908:    EF_AMDGPU_FEATURE_SRAMECC_V3 (0x200)
; SRAM-ECC-GFX908:    EF_AMDGPU_MACH_AMDGCN_GFX908 (0x30)
; SRAM-ECC-GFX908:  ]

; SRAM-ECC-GFX90A: Flags [
; SRAM-ECC-GFX90A:    EF_AMDGPU_FEATURE_SRAMECC_V3 (0x200)
; SRAM-ECC-GFX90A:    EF_AMDGPU_MACH_AMDGCN_GFX90A (0x3F)
; SRAM-ECC-GFX90A:  ]

define amdgpu_kernel void @elf_header() {
  ret void
}
