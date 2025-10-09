; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck --check-prefixes=ASM,ASM4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck --check-prefixes=ASM,ASM56 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck --check-prefixes=ASM,ASM56 %s

; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=4 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=5 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=6 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF6 %s

; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=4 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=5 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=4 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF6 %s

; ASM: .amdgcn_target  "amdgcn-amd-amdhsa--gfx900:xnack+"
; ASM:  amdhsa.target: 'amdgcn-amd-amdhsa--gfx900:xnack+'
; ASM:  amdhsa.version:
; ASM:     - 1
; ASM4:    - 1
; ASM56:   - 2

; ELF:      OS/ABI: AMDGPU_HSA (0x40)
; ELF4:      ABIVersion: 2
; ELF5:      ABIVersion: 3
; ELF6:      ABIVersion: 4
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
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
