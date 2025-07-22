; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brtc .+52
  brtc .+50
  brtc bar

bar:

; CHECK: brtc .Ltmp0+52+2  ; encoding: [0bAAAAA110,0b111101AA]
; CHECK: brtc .Ltmp1+50+2  ; encoding: [0bAAAAA110,0b111101AA]
; CHECK: brtc bar            ; encoding: [0bAAAAA110,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: d6 f4      brtc .+52
; INST-NEXT: ce f4      brtc .+50
; INST-NEXT: 06 f4      brtc .+0
