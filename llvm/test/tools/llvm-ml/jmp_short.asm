; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.code

t1:
  jmp short t1_label
  jmp SHORT t1_label
  JmP Short t1_label
  JMP SHORT t1_label
  mov eax, eax
t1_label:
  ret

; CHECK-LABEL: t1:
; CHECK-NEXT: jmp t1_label
; CHECK-NEXT: jmp t1_label
; CHECK-NEXT: jmp t1_label
; CHECK-NEXT: jmp t1_label
; CHECK-NEXT: mov eax, eax

end
