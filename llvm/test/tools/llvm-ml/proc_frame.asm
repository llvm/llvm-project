; RUN: llvm-ml64 -filetype=s %s /Fo - | FileCheck %s
; RUN: not llvm-ml64 -filetype=s %s /Fo /dev/null /DERR 2>&1 | FileCheck %s --check-prefix=CHECK-ERR --implicit-check-not=error:

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

t2 PROC PUBLIC FRAME
  .endprolog
  ret
t2 ENDP

; CHECK: .seh_proc t2
; CHECK: t2:
; CHECK: .seh_endprologue
; CHECK: ret
; CHECK: .seh_endproc

ifdef ERR
; CHECK-ERR: :[[# @LINE + 1]]:15: error: expected newline in 'PROC' directive
t3 PROC FRAME PUBLIC
; CHECK-ERR: :[[# @LINE + 1]]:4: error: endp outside of procedure block
t3 ENDP
endif

END
