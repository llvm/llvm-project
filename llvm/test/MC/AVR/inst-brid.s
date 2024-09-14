; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brid .+42
  brid .+62
  brid bar

bar:

; CHECK: brid (.Ltmp0+42)+2  ; encoding: [0bAAAAA111,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+42)+2, kind: fixup_7_pcrel
; CHECK: brid (.Ltmp1+62)+2  ; encoding: [0bAAAAA111,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+62)+2, kind: fixup_7_pcrel
; CHECK: brid bar            ; encoding: [0bAAAAA111,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: af f4      brid .+42
; INST-NEXT: ff f4      brid .+62
; INST-NEXT: 07 f4      brid .+0
