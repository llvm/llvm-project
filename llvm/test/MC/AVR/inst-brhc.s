; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brhc .+12
  brhc .+14
  brhc bar

bar:

; CHECK: brhc (.Ltmp0+12)+2  ; encoding: [0bAAAAA101,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+12)+2, kind: fixup_7_pcrel
; CHECK: brhc (.Ltmp1+14)+2  ; encoding: [0bAAAAA101,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1+14)+2, kind: fixup_7_pcrel
; CHECK: brhc bar            ; encoding: [0bAAAAA101,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 35 f4      brhc .+12
; INST-NEXT: 3d f4      brhc .+14
; INST-NEXT: 05 f4      brhc .+0
