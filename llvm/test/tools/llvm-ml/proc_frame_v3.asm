; RUN: llvm-ml64 -filetype=s /unwindv3 %s /Fo - | FileCheck %s

.code

; Verify @UnwindVersion reflects the /unwindv3 flag.
if @UnwindVersion ne 3
  .err <@UnwindVersion should be 3 when /unwindv3 is used>
endif

t1 PROC FRAME
  push2 r12, r13
  .push2reg r12, r13
  sub rsp, 32
  .allocstack 32
  .endprolog
  nop
  .beginepilog
  .freestack 32
  add rsp, 32
  .pop2reg r13, r12
  pop2 r13, r12
  .endepilog
  ret
t1 ENDP

; CHECK: .seh_proc t1

; CHECK: t1:
; CHECK: push2 r12, r13
; CHECK: .seh_push2regs r12, r13
; CHECK: sub rsp, 32
; CHECK: .seh_stackalloc 32
; CHECK: .seh_endprologue
; CHECK: nop
; CHECK: .seh_startepilogue
; CHECK: .seh_stackalloc 32
; CHECK: add rsp, 32
; CHECK: .seh_push2regs r12, r13
; CHECK: pop2 r13, r12
; CHECK: .seh_endepilogue
; CHECK: ret
; CHECK: .seh_endproc

; Test .popreg
t2 PROC FRAME
  push r12
  .pushreg r12
  .endprolog
  nop
  .beginepilog
  .popreg r12
  pop r12
  .endepilog
  ret
t2 ENDP

; CHECK: .seh_proc t2
; CHECK: .seh_pushreg r12
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_pushreg r12
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .freestack
t3 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  .endprolog
  nop
  .beginepilog
  .freestack 32
  add rsp, 32
  .popreg r12
  pop r12
  .endepilog
  ret
t3 ENDP

; CHECK: .seh_proc t3
; CHECK: .seh_stackalloc 32
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_stackalloc 32
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .restorereg
t4 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  mov [rsp], rbx
  .savereg rbx, 0
  .endprolog
  nop
  .beginepilog
  .restorereg rbx, 0
  mov rbx, [rsp]
  .freestack 32
  add rsp, 32
  .popreg r12
  pop r12
  .endepilog
  ret
t4 ENDP

; CHECK: .seh_proc t4
; CHECK: .seh_savereg rbx, 0
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_savereg rbx, 0
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .restorexmm128
t5 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  movaps [rsp], xmm6
  .savexmm128 xmm6, 0
  .endprolog
  nop
  .beginepilog
  .restorexmm128 xmm6, 0
  movaps xmm6, [rsp]
  .freestack 32
  add rsp, 32
  .popreg r12
  pop r12
  .endepilog
  ret
t5 ENDP

; CHECK: .seh_proc t5
; CHECK: .seh_savexmm xmm6, 0
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_savexmm xmm6, 0
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

; Test .unsetframe
t6 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  lea rbp, [rsp+16]
  .setframe rbp, 16
  .endprolog
  nop
  .beginepilog
  .unsetframe rbp, 16
  lea rsp, [rbp-16]
  .freestack 32
  add rsp, 32
  .popreg r12
  pop r12
  .endepilog
  ret
t6 ENDP

; CHECK: .seh_proc t6
; CHECK: .seh_setframe rbp, 16
; CHECK: .seh_endprologue
; CHECK: .seh_startepilogue
; CHECK: .seh_setframe rbp, 16
; CHECK: .seh_endepilogue
; CHECK: .seh_endproc

END
