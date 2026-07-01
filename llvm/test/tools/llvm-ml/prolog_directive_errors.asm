; RUN: not llvm-ml64 -filetype=s /unwindv3 %s /Fo - 2>&1 | FileCheck %s

.code

; Test A2256: prolog directives used after .endprolog
t1 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 32
  .allocstack 32
  .endprolog

  mov rax, 0

; CHECK: :[[#@LINE+1]]:3: error: prolog directive must be used inside a prolog
  .pushreg r13

; CHECK: :[[#@LINE+1]]:3: error: prolog directive must be used inside a prolog
  .push2reg r14, r15

; CHECK: :[[#@LINE+1]]:3: error: prolog directive must be used inside a prolog
  .pushframe

; CHECK: :[[#@LINE+1]]:3: error: prolog directive must be used inside a prolog
  .setframe rbp, 0

; CHECK: :[[#@LINE+1]]:3: error: prolog directive must be used inside a prolog
  .allocstack 16

; CHECK: :[[#@LINE+1]]:3: error: prolog directive must be used inside a prolog
  .savereg rbx, 0

; CHECK: :[[#@LINE+1]]:3: error: prolog directive must be used inside a prolog
  .savexmm128 xmm6, 0

  .beginepilog
  .freestack 32
  add rsp, 32
  .popreg r12
  pop r12
  .endepilog
  ret
t1 ENDP

END
