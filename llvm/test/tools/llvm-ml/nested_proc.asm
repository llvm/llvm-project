; RUN: llvm-ml -m32 -filetype=s %s /Fo - | FileCheck %s
; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.code

t1 PROC
  xor eax, eax
t1_nested PROC
  ret
t1_nested ENDP
t1 ENDP

; CHECK-LABEL: t1:
; CHECK: xor eax, eax
; CHECK: t1_nested:
; CHECK: ret

t2 PROC
  xor eax, eax
t2_nested PROC
  ret
T2_nEsTeD ENDP
t2 ENDP

; CHECK-LABEL: t2:
; CHECK: xor eax, eax
; CHECK: t2_nested:
; CHECK: ret

END
