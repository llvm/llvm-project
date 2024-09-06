; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brlt .+16
  brlt .+2
  brlt bar

bar:

; CHECK: brlt (.Ltmp0+16)+2  ; encoding: [0bAAAAA100,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+16)+2, kind: fixup_7_pcrel
; CHECK: brlt (.Ltmp1+2)+2   ; encoding: [0bAAAAA100,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+2)+2, kind: fixup_7_pcrel
; CHECK: brlt bar            ; encoding: [0bAAAAA100,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 44 f0    brlt .+16
; INST-NEXT: 0c f0    brlt .+2
; INST-NEXT: 04 f0    brlt .+0
