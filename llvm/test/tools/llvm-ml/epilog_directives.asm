; RUN: llvm-ml64 -filetype=s /unwindv3 %s /Fo - | FileCheck %s

.code

; Test .pushreg / .popreg
t1 PROC FRAME
  push r12
  .pushreg r12
  push r13
  .pushreg r13
  push rsi
  .pushreg rsi
  push rdi
  .pushreg rdi
  .endprolog
  mov rax, 0
  .beginepilog
  .popreg rdi
  pop rdi
  .popreg rsi
  pop rsi
  .popreg r13
  pop r13
  .popreg r12
  pop r12
  .endepilog
  ret
t1 ENDP

; CHECK: .seh_proc t1
; CHECK: push r12
; CHECK: .seh_pushreg r12
; CHECK: push r13
; CHECK: .seh_pushreg r13
; CHECK: push rsi
; CHECK: .seh_pushreg rsi
; CHECK: push rdi
; CHECK: .seh_pushreg rdi
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_pushreg rdi
; CHECK: pop rdi
; CHECK: .seh_pushreg rsi
; CHECK: pop rsi
; CHECK: .seh_pushreg r13
; CHECK: pop r13
; CHECK: .seh_pushreg r12
; CHECK: pop r12
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .allocstack / .freestack
t2 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 48
  .allocstack 48
  .endprolog
  mov rax, 0
  .beginepilog
  .freestack 48
  add rsp, 48
  .popreg r12
  pop r12
  .endepilog
  ret
t2 ENDP

; CHECK: .seh_proc t2
; CHECK: .seh_pushreg r12
; CHECK: .seh_stackalloc 48
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_stackalloc 48
; CHECK: .seh_pushreg r12
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .savereg / .restorereg
t3 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 48
  .allocstack 48
  mov [rsp], rbx
  .savereg rbx, 0
  mov [rsp+8], rsi
  .savereg rsi, 8
  .endprolog
  mov rax, 0
  .beginepilog
  .restorereg rsi, 8
  mov rsi, [rsp+8]
  .restorereg rbx, 0
  mov rbx, [rsp]
  .freestack 48
  add rsp, 48
  .popreg r12
  pop r12
  .endepilog
  ret
t3 ENDP

; CHECK: .seh_proc t3
; CHECK: .seh_savereg rbx, 0
; CHECK: .seh_savereg rsi, 8
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_savereg rsi, 8
; CHECK: .seh_savereg rbx, 0
; CHECK: .seh_stackalloc 48
; CHECK: .seh_pushreg r12
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .savexmm128 / .restorexmm128
t4 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 48
  .allocstack 48
  movaps [rsp], xmm6
  .savexmm128 xmm6, 0
  movaps [rsp+16], xmm7
  .savexmm128 xmm7, 16
  .endprolog
  mov rax, 0
  .beginepilog
  .restorexmm128 xmm7, 16
  movaps xmm7, [rsp+16]
  .restorexmm128 xmm6, 0
  movaps xmm6, [rsp]
  .freestack 48
  add rsp, 48
  .popreg r12
  pop r12
  .endepilog
  ret
t4 ENDP

; CHECK: .seh_proc t4
; CHECK: .seh_savexmm xmm6, 0
; CHECK: .seh_savexmm xmm7, 16
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_savexmm xmm7, 16
; CHECK: .seh_savexmm xmm6, 0
; CHECK: .seh_stackalloc 48
; CHECK: .seh_pushreg r12
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .setframe / .unsetframe
t5 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  lea rbp, [rsp+16]
  .setframe rbp, 16
  .endprolog
  mov rax, 0
  .beginepilog
  .unsetframe rbp, 16
  lea rsp, [rbp-16]
  .freestack 32
  add rsp, 32
  .popreg r12
  pop r12
  .endepilog
  ret
t5 ENDP

; CHECK: .seh_proc t5
; CHECK: .seh_setframe rbp, 16
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_setframe rbp, 16
; CHECK: .seh_stackalloc 32
; CHECK: .seh_pushreg r12
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .pushframe (interrupt handler)
t6 PROC FRAME
  .pushframe
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  .endprolog
  mov rax, 0
  .beginepilog
  .freestack 32
  add rsp, 32
  .popreg r12
  pop r12
  .endepilog
  iretq
t6 ENDP

; CHECK: .seh_proc t6
; CHECK: .seh_pushframe
; CHECK: .seh_pushreg r12
; CHECK: .seh_stackalloc 32
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_stackalloc 32
; CHECK: .seh_pushreg r12
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .pushframe code (interrupt handler with error code)
t7 PROC FRAME
  .pushframe code
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  .endprolog
  mov rax, 0
  .beginepilog
  .freestack 32
  add rsp, 32
  .popreg r12
  pop r12
  .endepilog
  iretq
t7 ENDP

; CHECK: .seh_proc t7
; CHECK: .seh_pushframe @code
; CHECK: .seh_endproc

; Test two epilogs
t8 PROC FRAME
  push r12
  .pushreg r12
  push rdi
  .pushreg rdi
  .endprolog
  mov rax, 0
  cmp rcx, 0
  je epilog2
  .beginepilog
  .popreg rdi
  pop rdi
  .popreg r12
  pop r12
  .endepilog
  ret
epilog2:
  .beginepilog
  .popreg rdi
  pop rdi
  .popreg r12
  pop r12
  .endepilog
  ret
t8 ENDP

; CHECK: .seh_proc t8
; CHECK: .seh_pushreg r12
; CHECK: .seh_pushreg rdi
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_pushreg rdi
; CHECK: .seh_pushreg r12
; CHECK: .seh_endepilogue
; CHECK: ret
; CHECK: .seh_startepilogue
; CHECK: .seh_pushreg rdi
; CHECK: .seh_pushreg r12
; CHECK: .seh_endepilogue
; CHECK: ret
; CHECK: .seh_endproc

END
