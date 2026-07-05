; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

OPTION pRoLoGuE:nOnE, EPILogue:None

.code

t1 PROC
  ret
t1 ENDP

; CHECK-LABEL: t1:
; CHECK-NOT: pop
; CHECK-NOT: push
; CHECK: {{^ *}}ret{{ *$}}

end

