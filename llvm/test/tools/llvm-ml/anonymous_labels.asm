; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.code

t1:
  jmp @F
  jmp @F
; CHECK-LABEL: t1:
; CHECK-NEXT: jmp [[TEMP1:[[:alpha:][:digit:]]+]]
; CHECK-NEXT: jmp [[TEMP1]]

@@:
  xor eax, eax
; CHECK: [[TEMP1]]:
; CHECK-NEXT: xor eax, eax

t2:
  jmp @B
  jmp @B
; CHECK-LABEL: t2:
; CHECK-NEXT: jmp [[TEMP1]]
; CHECK-NEXT: jmp [[TEMP1]]

t3:
  jmp @F
; CHECK-LABEL: t3:
; CHECK-NEXT: jmp [[TEMP2:[[:alpha:][:digit:]]+]]

@@:
  xor eax, eax
; CHECK: [[TEMP2]]:
; CHECK-NEXT: xor eax, eax

@@:
  xor eax, eax
; CHECK: [[TEMP3:[[:alpha:][:digit:]]+]]:
; CHECK-NEXT: xor eax, eax

t4:
  jmp @B
; CHECK-LABEL: t4:
; CHECK-NEXT: jmp [[TEMP3]]
