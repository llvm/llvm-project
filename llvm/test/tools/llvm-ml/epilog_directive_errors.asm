; RUN: not llvm-ml64 -filetype=s /unwindv3 %s /Fo - 2>&1 | FileCheck %s

.code

; Test A2255: epilog directives used outside .beginepilog/.endepilog
t1 PROC FRAME
  push r12
  .pushreg r12
  sub rsp, 64
  .allocstack 64
  lea rbp, [rsp]
  .setframe rbp, 0
  movaps [rsp+16], xmm6
  .savexmm128 xmm6, 16
  mov [rsp+32], rbx
  .savereg rbx, 32
  .endprolog

  mov rax, 0

; CHECK: :[[#@LINE+1]]:3: error: epilog directive must be used inside an epilog
  .popreg r12

; CHECK: :[[#@LINE+1]]:3: error: epilog directive must be used inside an epilog
  .pop2reg r12, r13

; CHECK: :[[#@LINE+1]]:3: error: epilog directive must be used inside an epilog
  .freestack 64

; CHECK: :[[#@LINE+1]]:3: error: epilog directive must be used inside an epilog
  .restorereg rbx, 32

; CHECK: :[[#@LINE+1]]:3: error: epilog directive must be used inside an epilog
  .restorexmm128 xmm6, 16

; CHECK: :[[#@LINE+1]]:3: error: epilog directive must be used inside an epilog
  .unsetframe rbp, 0

  ret
t1 ENDP

END
