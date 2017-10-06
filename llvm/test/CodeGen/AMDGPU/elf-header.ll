; RUN: llc -march=r600 -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=R600 --check-prefix=R600-OSABI-NONE %s
; RUN: llc -mtriple=r600-- -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=R600 --check-prefix=R600-OSABI-NONE %s
; RUN: llc -mtriple=r600-amd- -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=R600 --check-prefix=R600-OSABI-NONE %s
; RUN: llc -mtriple=r600-amd-unknown -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=R600 --check-prefix=R600-OSABI-NONE %s
; RUN: llc -mtriple=r600-unknown-unknown -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=R600 --check-prefix=R600-OSABI-NONE %s

; RUN: llc -march=amdgcn -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-NONE %s
; RUN: llc -mtriple=amdgcn-- -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-NONE %s
; RUN: llc -mtriple=amdgcn-amd- -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-NONE %s
; RUN: llc -mtriple=amdgcn-amd-unknown -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-NONE %s
; RUN: llc -mtriple=amdgcn-unknown-unknown -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-NONE %s

; RUN: llc -mtriple=amdgcn--amdhsa -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-HSA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-HSA %s
; RUN: llc -mtriple=amdgcn-unknown-amdhsa -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-HSA %s

; RUN: llc -mtriple=amdgcn--amdpal -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-PAL %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-PAL %s
; RUN: llc -mtriple=amdgcn-unknown-amdpal -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-PAL %s

; RUN: llc -mtriple=amdgcn--mesa3d -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-MESA3D %s
; RUN: llc -mtriple=amdgcn-amd-mesa3d -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-MESA3D %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck --check-prefix=GCN --check-prefix=GCN-OSABI-MESA3D %s

; R600: Format: ELF32-amdgpu
; R600: Arch: r600
; R600: AddressSize: 32bit
; GCN:  Format: ELF64-amdgpu
; GCN:  Arch: amdgcn
; GCN:  AddressSize: 64bit

; R600-OSABI-NONE:  OS/ABI: SystemV (0x0)
; GCN-OSABI-NONE:   OS/ABI: SystemV (0x0)
; GCN-OSABI-HSA:    OS/ABI: AMDGPU_HSA (0x40)
; GCN-OSABI-PAL:    OS/ABI: AMDGPU_PAL (0x41)
; GCN-OSABI-MESA3D: OS/ABI: AMDGPU_MESA3D (0x42)

; R600: Machine: EM_AMDGPU (0xE0)
; R600: Flags [ (0x1)
; R600:   EF_AMDGPU_ARCH_R600 (0x1)
; R600: ]
; GCN:  Machine: EM_AMDGPU (0xE0)
; GCN:  Flags [ (0x2)
; GCN:    EF_AMDGPU_ARCH_GCN (0x2)
; GCN:  ]

define amdgpu_kernel void @elf_header() {
  ret void
}
