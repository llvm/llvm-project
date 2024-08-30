; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brge .+50
  brge .+42
  brge bar

bar:

; CHECK: brge (.Ltmp0+50)+2  ; encoding: [0bAAAAA100,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+50)+2, kind: fixup_7_pcrel
; CHECK: brge (.Ltmp1+42)+2  ; encoding: [0bAAAAA100,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+42)+2, kind: fixup_7_pcrel
; CHECK: brge bar            ; encoding: [0bAAAAA100,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: cc f4      brge .+50
; INST-NEXT: ac f4      brge .+42
; INST-NEXT: 04 f4      brge .+0
