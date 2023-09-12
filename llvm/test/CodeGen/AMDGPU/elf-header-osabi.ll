; RUN: llc -filetype=obj -mtriple=amdgcn -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NONE %s
; RUN: llc -filetype=obj -mtriple=amdgcn-amd- -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NONE %s
; RUN: llc -filetype=obj -mtriple=amdgcn-amd-unknown -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=NONE %s
; RUN: llc -filetype=obj -mtriple=amdgcn--amdhsa -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA %s
; RUN: llc -filetype=obj -mtriple=amdgcn-amd-amdhsa -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA %s
; RUN: llc -filetype=obj -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=HSA %s
; RUN: llc -filetype=obj -mtriple=amdgcn--amdpal -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=PAL %s
; RUN: llc -filetype=obj -mtriple=amdgcn-amd-amdpal -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=PAL %s
; RUN: llc -filetype=obj -mtriple=amdgcn-unknown-amdpal -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=PAL %s
; RUN: llc -filetype=obj -mtriple=amdgcn--mesa3d -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=MESA3D %s
; RUN: llc -filetype=obj -mtriple=amdgcn-amd-mesa3d -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=MESA3D %s
; RUN: llc -filetype=obj -mtriple=amdgcn-unknown-mesa3d -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=MESA3D %s

; NONE:   OS/ABI: SystemV       (0x0)
; HSA:    OS/ABI: AMDGPU_HSA    (0x40)
; HSA:    ABIVersion: 3
; PAL:    OS/ABI: AMDGPU_PAL    (0x41)
; PAL:    ABIVersion: 0
; MESA3D: OS/ABI: AMDGPU_MESA3D (0x42)
; MESA3D: ABIVersion: 0

define amdgpu_kernel void @elf_header() {
  ret void
}
