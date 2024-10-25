; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brvc .-28
  brvc .-62
  brvc bar

bar:

; CHECK: brvc (.Ltmp0-28)+2  ; encoding: [0bAAAAA011,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0-28)+2, kind: fixup_7_pcrel
; CHECK: brvc (.Ltmp1-62)+2  ; encoding: [0bAAAAA011,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1-62)+2, kind: fixup_7_pcrel
; CHECK: brvc bar            ; encoding: [0bAAAAA011,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 93 f7      brvc .-28
; INST-NEXT: 0b f7      brvc .-62
; INST-NEXT: 03 f4      brvc .+0
