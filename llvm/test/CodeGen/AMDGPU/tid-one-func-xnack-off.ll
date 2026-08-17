; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa | FileCheck --check-prefixes=ASM,ASM4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa | FileCheck --check-prefixes=ASM,ASM56 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa | FileCheck --check-prefixes=ASM,ASM56 %s

; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=4 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=5 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=6 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF6 %s

; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=4 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=5 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=4 --filetype=obj | llvm-readobj --file-headers - | FileCheck --check-prefixes=ELF,ELF6 %s

; ASM: .amdgcn_target  "amdgpu9.00-amd-amdhsa-unknown-gfx900:xnack-"
; ASM:  amdhsa.target: 'amdgpu9.00-amd-amdhsa-unknown-gfx900:xnack-'
; ASM:  amdhsa.version:
; ASM:     - 1
; ASM4:    - 1
; ASM56:   - 2

; ELF:      OS/ABI: AMDGPU_HSA (0x40)
; ELF4:      ABIVersion: 2
; ELF5:      ABIVersion: 3
; ELF6:      ABIVersion: 4
; ELF:      Flags [ (0x22C)
; ELF-NEXT:   EF_AMDGPU_FEATURE_XNACK_OFF_V4 (0x200)
; ELF-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX900   (0x2C)
; ELF-NEXT: ]

define void @func0() #0 {
entry:
  ret void
}

attributes #0 = { "target-features"="-xnack" }

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
!1 = !{i32 1, !"amdgpu.xnack", i32 0}
