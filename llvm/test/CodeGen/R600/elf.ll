; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs -filetype=obj | llvm-readobj -s -symbols - | FileCheck --check-prefix=ELF-CHECK %s
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs -o - | FileCheck --check-prefix=CONFIG-CHECK %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs -filetype=obj | llvm-readobj -s -symbols - | FileCheck --check-prefix=ELF-CHECK %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs -o - | FileCheck --check-prefix=CONFIG-CHECK %s

; ELF-CHECK: Format: ELF32
; ELF-CHECK: Name: .AMDGPU.config
; ELF-CHECK: Type: SHT_PROGBITS

; ELF-CHECK: Symbol {
; ELF-CHECK: Name: test
; ELF-CHECK: Binding: Global

; CONFIG-CHECK: .align 256
; CONFIG-CHECK: test:
; CONFIG-CHECK: .section .AMDGPU.config
; CONFIG-CHECK-NEXT: .long   45096
; CONFIG-CHECK-NEXT: .long   0
define void @test(i32 %p) #0 {
   %i = add i32 %p, 2
   %r = bitcast i32 %i to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %r, float %r, float %r, float %r)
   ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" } ; Pixel Shader
