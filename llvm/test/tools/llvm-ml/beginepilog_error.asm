; RUN: not llvm-ml64 -filetype=s /unwindv3 %s /Fo - 2>&1 | FileCheck %s

.code

; Test A2265: .beginepilog inside .beginepilog (before .endepilog)
t1 PROC FRAME
  push r10
  .pushreg r10
  .endprolog
  .beginepilog
  .popreg r10
  pop r10
; CHECK: .beginepilog must come after .endprolog or .endepilog
  .beginepilog
  .endepilog
  ret
t1 ENDP

END
