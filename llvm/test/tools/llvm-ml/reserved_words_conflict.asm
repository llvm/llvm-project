; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.code

t1:
call dword ptr [eax]

; CHECK-LABEL: t1:
; CHECK-NEXT: call

t2 dword 5

; CHECK-LABEL: t2:
; CHECK-NEXT: .long 5

END
