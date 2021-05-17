; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=XNACK-GFX801 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx801 -mattr=+xnack < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=XNACK-GFX801 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx802 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-XNACK-GFX802 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx802 -mattr=-xnack < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NO-XNACK-GFX802 %s

; XNACK-GFX801:      Flags [
; XNACK-GFX801-NEXT:   EF_AMDGPU_FEATURE_XNACK_V3   (0x100)
; XNACK-GFX801-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX801 (0x28)
; XNACK-GFX801-NEXT: ]

; NO-XNACK-GFX802:      Flags [
; NO-XNACK-GFX802-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX802 (0x29)
; NO-XNACK-GFX802-NEXT: ]

define amdgpu_kernel void @elf_header() {
  ret void
}
