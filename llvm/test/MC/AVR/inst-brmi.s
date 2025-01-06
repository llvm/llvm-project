; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brmi .+66
  brmi .+58
  brmi bar

bar:

; CHECK: brmi (.Ltmp0+66)+2  ; encoding: [0bAAAAA010,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+66)+2, kind: fixup_7_pcrel
; CHECK: brmi (.Ltmp1+58)+2  ; encoding: [0bAAAAA010,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+58)+2, kind: fixup_7_pcrel
; CHECK: brmi bar            ; encoding: [0bAAAAA010,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 0a f1      brmi .+66
; INST-NEXT: ea f0      brmi .+58
; INST-NEXT: 02 f0      brmi .+0
