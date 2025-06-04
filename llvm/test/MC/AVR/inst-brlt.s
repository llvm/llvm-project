; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brlt .+16
  brlt .+2
  brlt bar

bar:

; CHECK: brlt .Ltmp0+16+2  ; encoding: [0bAAAAA100,0b111100AA]
; CHECK: brlt .Ltmp1+2+2   ; encoding: [0bAAAAA100,0b111100AA]
; CHECK: brlt bar            ; encoding: [0bAAAAA100,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: 44 f0    brlt .+16
; INST-NEXT: 0c f0    brlt .+2
; INST-NEXT: 04 f0    brlt .+0
