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
; CHECK: :[[#@LINE+1]]:3: error: .beginepilog must come after .endprolog or .endepilog
  .beginepilog
  .endepilog
  ret
t1 ENDP

; .beginepilog before the prolog has ended (.endprolog) is also rejected.
t2 PROC FRAME
  push r10
  .pushreg r10
; CHECK: :[[#@LINE+1]]:3: error: .beginepilog must come after .endprolog or .endepilog
  .beginepilog
  .popreg r10
  pop r10
  .endepilog
  ret
t2 ENDP

; .endepilog without a matching .beginepilog is rejected.
t3 PROC FRAME
  push r10
  .pushreg r10
  .endprolog
  pop r10
; CHECK: :[[#@LINE+1]]:3: error: epilog directive must be used inside an epilog
  .endepilog
  ret
t3 ENDP

END
