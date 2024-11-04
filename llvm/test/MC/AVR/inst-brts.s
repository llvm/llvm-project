; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brts .+18
  brts .+22
  brts bar

bar:

; CHECK: brts (.Ltmp0+18)+2  ; encoding: [0bAAAAA110,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+18)+2, kind: fixup_7_pcrel
; CHECK: brts (.Ltmp1+22)+2  ; encoding: [0bAAAAA110,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+22)+2, kind: fixup_7_pcrel
; CHECK: brts bar            ; encoding: [0bAAAAA110,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 4e f0      brts .+18
; INST-NEXT: 5e f0      brts .+22
; INST-NEXT: 06 f0      brts .+0
