; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brne .+10
  brne .+2
  brbc 1, .+10
  brbc 1, bar

bar:

; CHECK: brne    .Ltmp0+10+2     ; encoding: [0bAAAAA001,0b111101AA]
; CHECK: brne    .Ltmp1+2+2      ; encoding: [0bAAAAA001,0b111101AA]
; CHECK: brbc    1, .Ltmp2+10+2  ; encoding: [0bAAAAA001,0b111101AA]
; CHECK: brbc    1, bar            ; encoding: [0bAAAAA001,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: 29 f4      brne .+10
; INST-NEXT: 09 f4      brne .+2
; INST-NEXT: 29 f4      brne .+10
; INST-NEXT: 01 f4      brne .+0
