; RUN: llvm-ml %s /Zs /Fo - | FileCheck %s

.code

t1 PROC
  ECHO Testing!
  ret
t1 ENDP

; check for the .text symbol (appears in both object files & .s output)
; CHECK-NOT: .text

; CHECK: Testing!

; check for the .text symbol (appears in both object files & .s output)
; CHECK-NOT: .text

end
