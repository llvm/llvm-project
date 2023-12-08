; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=-sse2 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=X64

; PR16133 - we must treat XMM registers as v4f32 as SSE1 targets don't permit other vector types.

define void @nop() nounwind {
; X32-LABEL: nop:
; X32:       # %bb.0:
; X32-NEXT:    pushl %ebp
; X32-NEXT:    movl %esp, %ebp
; X32-NEXT:    andl $-16, %esp
; X32-NEXT:    subl $32, %esp
; X32-NEXT:    #APP
; X32-NEXT:    #NO_APP
; X32-NEXT:    movaps %xmm0, (%esp)
; X32-NEXT:    movl %ebp, %esp
; X32-NEXT:    popl %ebp
; X32-NEXT:    retl
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
