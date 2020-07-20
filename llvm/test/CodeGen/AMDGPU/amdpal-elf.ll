; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=kaveri -filetype=obj -mattr=-code-object-v3 | llvm-readobj -symbols -s -sd - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=kaveri -mattr=-code-object-v3 | llvm-mc -filetype=obj -triple amdgcn--amdpal -mcpu=kaveri -mattr=-code-object-v3 | llvm-readobj -symbols -s -sd - | FileCheck %s --check-prefix=ELF
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1010 -mattr=+WavefrontSize32,-WavefrontSize64,-code-object-v3 | FileCheck --check-prefix=GFX10-W32 %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1010 -mattr=-WavefrontSize32,+WavefrontSize64,-code-object-v3 | FileCheck --check-prefix=GFX10-W64 %s

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

; GFX10-W32: NumSGPRsForWavesPerEU: 4
; GFX10-W32: NumVGPRsForWavesPerEU: 3
; GFX10-W64: NumSGPRsForWavesPerEU: 2
; GFX10-W64: NumVGPRsForWavesPerEU: 3

define amdgpu_kernel void @simple(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}
