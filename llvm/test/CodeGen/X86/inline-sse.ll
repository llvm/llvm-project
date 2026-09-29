; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse | FileCheck %s --check-prefix=X86
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=-sse2 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=X64

; PR16133 - we must treat XMM registers as v4f32 as SSE1 targets don't permit other vector types.

define void @nop() nounwind {
; X86-LABEL: nop:
; X86:       # %bb.0:
; X86-NEXT:    pushl %ebp
; X86-NEXT:    movl %esp, %ebp
; X86-NEXT:    andl $-16, %esp
; X86-NEXT:    subl $32, %esp
; X86-NEXT:    #APP
; X86-NEXT:    #NO_APP
; X86-NEXT:    movaps %xmm0, (%esp)
; X86-NEXT:    movl %ebp, %esp
; X86-NEXT:    popl %ebp
; X86-NEXT:    retl
;
; X64-LABEL: nop:
; X64:       # %bb.0:
; X64-NEXT:    #APP
; X64-NEXT:    #NO_APP
; X64-NEXT:    movaps %xmm0, -{{[0-9]+}}(%rsp)
; X64-NEXT:    retq
  %1 = alloca <4 x float>, align 16
  %2 = call <4 x float> asm "", "=x,~{dirflag},~{fpsr},~{flags}"()
  store <4 x float> %2, ptr %1, align 16
  ret void
}
