; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brhs .-66
  brhs .+14
  brhs bar

bar:

; CHECK: brhs (.Ltmp0-66)+2  ; encoding: [0bAAAAA101,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0-66)+2, kind: fixup_7_pcrel
; CHECK: brhs (.Ltmp1+14)+2  ; encoding: [0bAAAAA101,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+14)+2, kind: fixup_7_pcrel
; CHECK: brhs bar            ; encoding: [0bAAAAA101,0b111100AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: fd f2      brhs .-66
; INST-NEXT: 3d f0      brhs .+14
; INST-NEXT: 05 f0      brhs .+0
