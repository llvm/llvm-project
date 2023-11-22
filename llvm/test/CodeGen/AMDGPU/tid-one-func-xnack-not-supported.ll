; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 < %s | FileCheck --check-prefixes=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --filetype=obj < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF %s

; ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx700"
; ASM:  amdhsa.target: amdgcn-amd-amdhsa--gfx700
; ASM:  amdhsa.version:
; ASM:    - 1
; ASM:    - 2

; ELF:      OS/ABI: AMDGPU_HSA (0x40)
; ELF:      ABIVersion: 3
; ELF:      Flags [ (0x22)
; ELF-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX700 (0x22)
; ELF-NEXT: ]

define void @func0() {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 500}
