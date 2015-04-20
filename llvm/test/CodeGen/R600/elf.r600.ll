; RUN: llc < %s -march=r600 -mcpu=redwood -filetype=obj | llvm-readobj -s - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -march=r600 -mcpu=redwood -o - | FileCheck --check-prefix=CONFIG %s

; ELF: Format: ELF32
; ELF: Name: .AMDGPU.config

; CONFIG: .section .AMDGPU.config
; CONFIG-NEXT: .long   166100
; CONFIG-NEXT: .long   2
; CONFIG-NEXT: .long   165900
; CONFIG-NEXT: .long   0
define void @test(float addrspace(1)* %out, i32 %p) {
   %i = add i32 %p, 2
   %r = bitcast i32 %i to float
   store float %r, float addrspace(1)* %out
   ret void
}
