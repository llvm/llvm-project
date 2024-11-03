; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefixes=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --filetype=obj < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF %s

; ASM: .amdgcn_target  "amdgcn-amd-amdhsa--gfx900:xnack+"
; ASM:  amdhsa.target: 'amdgcn-amd-amdhsa--gfx900:xnack+'
; ASM:  amdhsa.version:
; ASM:    - 1
; ASM:    - 1

; ELF:      OS/ABI: AMDGPU_HSA (0x40)
; ELF:      ABIVersion: 2
; ELF:      Flags [ (0x32C)
; ELF-NEXT:   EF_AMDGPU_FEATURE_XNACK_ON_V4 (0x300)
; ELF-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX900  (0x2C)
; ELF-NEXT: ]

define void @func0() {
entry:
  ret void
}

define void @func1() #0 {
entry:
  ret void
}

define void @func2() {
entry:
  ret void
}

attributes #0 = { "target-features"="+xnack" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
