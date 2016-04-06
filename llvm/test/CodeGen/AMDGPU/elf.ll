; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs -filetype=obj | llvm-readobj -s -symbols -file-headers - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs -o - | FileCheck --check-prefix=CONFIG --check-prefix=TYPICAL %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs -filetype=obj | llvm-readobj -s -symbols -file-headers - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs -o - | FileCheck --check-prefix=CONFIG --check-prefix=TONGA %s
; RUN: llc < %s -march=amdgcn -mcpu=carrizo -verify-machineinstrs -filetype=obj | llvm-readobj -s -symbols -file-headers - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -march=amdgcn -mcpu=carrizo -verify-machineinstrs -o - | FileCheck --check-prefix=CONFIG --check-prefix=TYPICAL %s

; Test that we don't try to produce a COFF file on windows
; RUN: llc < %s -mtriple=amdgcn-pc-mingw -mcpu=SI -verify-machineinstrs -filetype=obj | llvm-readobj -s -symbols -file-headers - | FileCheck --check-prefix=ELF %s

; ELF: Format: ELF64
; ELF: OS/ABI: AMDGPU_HSA (0x40)
; ELF: Machine: EM_AMDGPU (0xE0)
; ELF: Name: .AMDGPU.config
; ELF: Type: SHT_PROGBITS

; ELF: Symbol {
; ELF: Name: test
; ELF: Binding: Global

; CONFIG: .section .AMDGPU.config
; CONFIG-NEXT: .long   45096
; TYPICAL-NEXT: .long   0
; TONGA-NEXT: .long   576
; CONFIG: .p2align 8
; CONFIG: test:
define amdgpu_ps void @test(i32 %p) {
   %i = add i32 %p, 2
   %r = bitcast i32 %i to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %r, float %r, float %r, float %r)
   ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
