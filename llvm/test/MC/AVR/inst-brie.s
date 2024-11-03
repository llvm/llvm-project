; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brie .+20
  brie .+40
  brie bar

bar:

; CHECK: brie (.Ltmp0+20)+2  ; encoding: [0bAAAAA111,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+20)+2, kind: fixup_7_pcrel
; CHECK: brie (.Ltmp1+40)+2  ; encoding: [0bAAAAA111,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+40)+2, kind: fixup_7_pcrel
; CHECK: brie bar            ; encoding: [0bAAAAA111,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 57 f0      brie .+20
; INST-NEXT: a7 f0      brie .+40
; INST-NEXT: 07 f0      brie .+0
