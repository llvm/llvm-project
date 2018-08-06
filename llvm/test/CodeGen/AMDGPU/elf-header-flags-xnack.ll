; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx801 -mattr=-xnack < %s | llvm-readobj -file-headers - | FileCheck --check-prefixes=NO-XNACK-GFX801 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx802 -mattr=+xnack < %s | llvm-readobj -file-headers - | FileCheck --check-prefixes=XNACK-GFX802 %s

; NO-XNACK-GFX801:      Flags [
; NO-XNACK-GFX801-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX801 (0x28)
; NO-XNACK-GFX801-NEXT: ]

; XNACK-GFX802:      Flags [
; XNACK-GFX802-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX802 (0x29)
; XNACK-GFX802-NEXT:   EF_AMDGPU_XNACK              (0x100)
; XNACK-GFX802-NEXT: ]

define amdgpu_kernel void @elf_header() {
  ret void
}
