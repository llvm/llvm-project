; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brpl .-12
  brpl .+18
  brpl bar

bar:

; CHECK: brpl .Ltmp0-12+2  ; encoding: [0bAAAAA010,0b111101AA]
; CHECK: brpl .Ltmp1+18+2  ; encoding: [0bAAAAA010,0b111101AA]
; CHECK: brpl bar            ; encoding: [0bAAAAA010,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: d2 f7      brpl .-12
; INST-NEXT: 4a f4      brpl .+18
; INST-NEXT: 02 f4      brpl .+0
