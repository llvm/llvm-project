; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx906 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx906 -mattr=-sramecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx906 -mattr=+sramecc < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-GFX906 %s
; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx906 -mattr=+sramecc,+xnack < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=SRAM-ECC-XNACK-GFX906 %s

; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx908 < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX908 %s
; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx908 -mattr=+sramecc < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX908 %s

; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx90a < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX90A %s
; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx90a < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX90A %s

; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx942 < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX942 %s
; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx942 -mattr=+sramecc < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX942 %s

; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx950 < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX950 %s
; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx950 -mattr=+sramecc < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX950 %s

; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx1250 < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX1250 %s
; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx1250 -mattr=+sramecc < %s | llvm-readobj --file-header - | FileCheck --check-prefix=SRAM-ECC-GFX1250 %s

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

; SRAM-ECC-GFX942: Flags [
; SRAM-ECC-GFX942:    EF_AMDGPU_FEATURE_SRAMECC_V3 (0x200)
; SRAM-ECC-GFX942:    EF_AMDGPU_MACH_AMDGCN_GFX942 (0x4C)
; SRAM-ECC-GFX942:  ]

; SRAM-ECC-GFX950: Flags [
; SRAM-ECC-GFX950:    EF_AMDGPU_FEATURE_SRAMECC_V3 (0x200)
; SRAM-ECC-GFX950:    EF_AMDGPU_MACH_AMDGCN_GFX950 (0x4F)
; SRAM-ECC-GFX950:  ]

; SRAM-ECC-GFX1250: Flags [
; SRAM-ECC-GFX1250:    EF_AMDGPU_FEATURE_SRAMECC_V3 (0x200)
; SRAM-ECC-GFX1250:    EF_AMDGPU_MACH_AMDGCN_GFX1250 (0x49)
; SRAM-ECC-GFX1250:  ]

define amdgpu_kernel void @elf_header() {
  ret void
}
