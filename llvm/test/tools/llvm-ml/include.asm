; RUN: llvm-ml -filetype=s %s /I %S /Fo - | FileCheck %s

include included.inc

.code

t1:
mov eax, Const

; CHECK-LABEL: t1:
; CHECK-NEXT: mov eax, 8

t2:
push_pop ebx

; CHECK-LABEL: t2:
; CHECK-NEXT: push ebx
; CHECK-NEXT: pop ebx

end
