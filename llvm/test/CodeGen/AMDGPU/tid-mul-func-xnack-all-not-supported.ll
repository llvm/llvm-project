; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 | FileCheck --check-prefixes=ASM,ASM4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 | FileCheck --check-prefixes=ASM,ASM5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=4 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=5 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF5 %s

; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=4 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=5 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF5 %s

; ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx700"
; ASM:  amdhsa.target: amdgcn-amd-amdhsa--gfx700
; ASM:  amdhsa.version:
; ASM:    - 1
; ASM4:   - 1
; ASM5:   - 2

; ELF:      OS/ABI: AMDGPU_HSA (0x40)
; ELF4:      ABIVersion: 2
; ELF5:      ABIVersion: 3
; ELF:      Flags [ (0x22)
; ELF-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX700 (0x22)
; ELF-NEXT: ]

define void @func0() {
entry:
  ret void
}

define void @func1() {
entry:
  ret void
}

define void @func2() {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 CODE_OBJECT_VERSION}
