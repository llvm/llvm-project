; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=kaveri -filetype=obj | llvm-readobj -S --sd --syms - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=kaveri | llvm-mc -filetype=obj -triple amdgcn--amdpal -mcpu=kaveri | llvm-readobj -S --sd --syms - | FileCheck %s --check-prefix=ELF
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1010 -mattr=+wavefrontsize32 | FileCheck --check-prefix=GFX10 %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1010 -mattr=+wavefrontsize64 | FileCheck --check-prefix=GFX10 %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1100 -mattr=+wavefrontsize32 | FileCheck --check-prefix=GFX11W32 %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1100 -mattr=+wavefrontsize64 | FileCheck --check-prefix=GFX11W64 %s

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

; GFX10: NumSGPRsForWavesPerEU: 12
; GFX10: NumVGPRsForWavesPerEU: 3

; Wave32 and 64 behave differently due to the UserSGPRInit16Bug,
; which only affects Wave32.
; GFX11W32: NumSGPRsForWavesPerEU: 16
; GFX11W32: NumVGPRsForWavesPerEU: 1

; GFX11W64: NumSGPRsForWavesPerEU: 11
; GFX11W64: NumVGPRsForWavesPerEU: 1

define amdgpu_kernel void @simple(ptr addrspace(1) %out) {
entry:
  store i32 0, ptr addrspace(1) %out
  ret void
}
