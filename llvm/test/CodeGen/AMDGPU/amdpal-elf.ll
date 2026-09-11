; RUN: llc < %s -mtriple=amdgpu7.00--amdpal -filetype=obj | llvm-readobj -S --sd --syms - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgpu7.00--amdpal | llvm-mc -filetype=obj -triple amdgpu7.00--amdpal -mcpu=kaveri | llvm-readobj -S --sd --syms - | FileCheck %s --check-prefix=ELF
; RUN: llc < %s -mtriple=amdgpu10.10--amdpal -mattr=+wavefrontsize32 | FileCheck --check-prefix=GFX10 %s
; RUN: llc < %s -mtriple=amdgpu10.10--amdpal -mattr=+wavefrontsize64 | FileCheck --check-prefix=GFX10 %s
; RUN: llc < %s -mtriple=amdgpu11.00--amdpal -mattr=+wavefrontsize32 | FileCheck --check-prefix=GFX10 %s
; RUN: llc < %s -mtriple=amdgpu11.00--amdpal -mattr=+wavefrontsize64 | FileCheck --check-prefix=GFX10 %s

; ELF: Section {
; ELF: Name: .text
; ELF: Type: SHT_PROGBITS (0x1)
; ELF: Flags [ (0x6)
; ELF: SHF_ALLOC (0x2)
; ELF: SHF_EXECINSTR (0x4)
; ELF: }

; ELF: SHT_NOTE
; ELF: Flags [ (0x0)
; ELF: ]

; ELF: Symbol {
; ELF: Name: simple
; ELF: Size: 36
; ELF: Section: .text (0x2)
; ELF: }

; GFX10: NumSGPRsForWavesPerEU: 6
; GFX10: NumVGPRsForWavesPerEU: 1

define amdgpu_kernel void @simple(ptr addrspace(1) %out) {
entry:
  store i32 0, ptr addrspace(1) %out
  ret void
}
