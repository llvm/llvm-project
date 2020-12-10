# RUN: llvm-ml -m32 -filetype=s %s /Fo - | FileCheck %s
# RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.code

t1 PROC
  ret
t1 ENDP

; CHECK: t1:
; CHECK: ret

END
