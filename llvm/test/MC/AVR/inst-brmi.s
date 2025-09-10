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

; CHECK: brmi .Ltmp0+66+2  ; encoding: [0bAAAAA010,0b111100AA]
; CHECK: brmi .Ltmp1+58+2  ; encoding: [0bAAAAA010,0b111100AA]
; CHECK: brmi bar            ; encoding: [0bAAAAA010,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: fa f3      brmi .-2
; INST-NEXT: VR_7_PCREL .text+0x44
; INST-NEXT: fa f3      brmi .-2
; INST-NEXT: VR_7_PCREL .text+0x3e
; INST-NEXT: fa f3      brmi .-2
; INST-NEXT: VR_7_PCREL .text+0x6
