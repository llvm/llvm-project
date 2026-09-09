; RUN: llc -filetype=obj -mtriple=amdgpu8.01 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NONE %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-amd- < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NONE %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-amd-unknown < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NONE %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01--amdhsa --amdhsa-code-object-version=4 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA4 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-amd-amdhsa --amdhsa-code-object-version=4 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA4 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-unknown-amdhsa --amdhsa-code-object-version=4 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA4 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01--amdhsa --amdhsa-code-object-version=5 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA5 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-amd-amdhsa --amdhsa-code-object-version=5 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA5 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-unknown-amdhsa --amdhsa-code-object-version=5 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA5 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01--amdhsa --amdhsa-code-object-version=6 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA6 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-amd-amdhsa --amdhsa-code-object-version=6 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA6 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-unknown-amdhsa --amdhsa-code-object-version=6 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA,HSA6 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01--amdpal < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=PAL %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-amd-amdpal < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=PAL %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-unknown-amdpal < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=PAL %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01--mesa3d < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=MESA3D %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-amd-mesa3d < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=MESA3D %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01-unknown-mesa3d < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=MESA3D %s

; NONE:   OS/ABI: SystemV       (0x0)
; HSA:    OS/ABI: AMDGPU_HSA    (0x40)
; HSA4:    ABIVersion: 2
; HSA5:    ABIVersion: 3
; HSA6:    ABIVersion: 4
; PAL:    OS/ABI: AMDGPU_PAL    (0x41)
; PAL:    ABIVersion: 0
; MESA3D: OS/ABI: AMDGPU_MESA3D (0x42)
; MESA3D: ABIVersion: 0

define amdgpu_kernel void @elf_header() {
  ret void
}
