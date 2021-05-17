; RUN: not llvm-ml %s /Zs /Fo - 2>&1 | FileCheck %s

.code

t1 PROC
  blah
  ret
t1 ENDP

; check for the .text symbol (appears in both object files & .s output)
; CHECK-NOT: .text

; CHECK: error: invalid instruction mnemonic 'blah'

; check for the .text symbol (appears in both object files & .s output)
; CHECK-NOT: .text

end
