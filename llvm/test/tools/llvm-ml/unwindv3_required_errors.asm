; RUN: not llvm-ml64 -filetype=s %s /Fo - 2>&1 | FileCheck %s

; These directives require unwind v3, but this file is assembled WITHOUT the
; /unwindv3 flag (so the default unwind version is 1). Each use must be
; diagnosed rather than silently mis-encoded.

.code

; .push2reg / .pop2reg map to UOP_Push2, which only exists in v3.
t1 PROC FRAME
  push2 r12, r13
  .push2reg r12, r13
; CHECK: :[[#@LINE-1]]:3: error: .seh_push2regs is only supported for unwind v3
  .endprolog
  ret
t1 ENDP

; Epilog unwind codes only exist in v3.
t2 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  .endprolog
  nop
  .beginepilog
  .freestack 32
; CHECK: :[[#@LINE-1]]:3: error: .seh_stackalloc inside epilog requires unwind v3
  add rsp, 32
  .popreg r12
; CHECK: :[[#@LINE-1]]:3: error: .seh_pushreg inside epilog requires unwind v3
  pop r12
  .endepilog
  ret
t2 ENDP

; Extended registers (r16-r31 / xmm16-xmm31) do not fit in the 4-bit register
; field of a v1/v2 unwind code, so they must be diagnosed rather than truncated
; to a low register.
t3 PROC FRAME
  push r16
  .pushreg r16
; CHECK: :[[#@LINE-1]]:3: error: .seh_pushreg with an extended register requires unwind v3
  .endprolog
  ret
t3 ENDP

t4 PROC FRAME
  sub rsp, 32
  .allocstack 32
  .savereg r16, 0
; CHECK: :[[#@LINE-1]]:3: error: .seh_savereg with an extended register requires unwind v3
  .savexmm128 xmm16, 16
; CHECK: :[[#@LINE-1]]:3: error: .seh_savexmm with an extended register requires unwind v3
  .setframe r17, 0
; CHECK: :[[#@LINE-1]]:3: error: .seh_setframe with an extended register requires unwind v3
  .endprolog
  ret
t4 ENDP

END
