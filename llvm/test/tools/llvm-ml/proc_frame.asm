; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.code

t1 PROC FRAME
  push rbp
  .pushreg rbp
  mov rbp, rsp
  .setframe rbp, 0
  pushfq
  .allocstack 8
  .endprolog
  ret
t1 ENDP

; CHECK: .seh_proc t1

; CHECK: t1:
; CHECK: push rbp
; CHECK: .seh_pushreg rbp
; CHECK: mov rbp, rsp
; CHECK: .seh_setframe rbp, 0
; CHECK: pushfq
; CHECK: .seh_stackalloc 8
; CHECK: .seh_endprologue
; CHECK: ret
; CHECK: .seh_endproc

END
