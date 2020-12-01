; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=kaveri -filetype=obj | llvm-readobj -symbols -s -sd - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=kaveri | llvm-mc -filetype=obj -triple amdgcn--amdpal -mcpu=kaveri | llvm-readobj -symbols -s -sd - | FileCheck %s --check-prefix=ELF
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 | FileCheck --check-prefix=GFX10 %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 | FileCheck --check-prefix=GFX10 %s

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

; GFX10: NumSGPRsForWavesPerEU: 2
; GFX10: NumVGPRsForWavesPerEU: 1

define amdgpu_kernel void @simple(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}
