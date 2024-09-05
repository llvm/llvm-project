; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brvs .+18
  brvs .+32
  brvs bar

bar:

; CHECK: brvs (.Ltmp0+18)+2  ; encoding: [0bAAAAA011,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+18)+2, kind: fixup_7_pcrel
; CHECK: brvs (.Ltmp1+32)+2  ; encoding: [0bAAAAA011,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+32)+2, kind: fixup_7_pcrel
; CHECK: brvs bar            ; encoding: [0bAAAAA011,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 4b f0      brvs .+18
; INST-NEXT: 83 f0      brvs .+32
; INST-NEXT: 03 f0      brvs .+0
