; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=sram < %s | llvm-objdump --no-print-imm-hex -dr --mattr=sram - | FileCheck --check-prefix=CHECK-INST %s

foo:
  ldi r16, 241
  ldi r29, 190
  ldi r22, 172
  ldi r27, 92
  ldi r21, SYMBOL+3

; CHECK: ldi r16, 241                 ; encoding: [0x01,0xef]
; CHECK: ldi r29, 190                 ; encoding: [0xde,0xeb]
; CHECK: ldi r22, 172                 ; encoding: [0x6c,0xea]
; CHECK: ldi r27, 92                  ; encoding: [0xbc,0xe5]

; CHECK: ldi r21, SYMBOL+3            ; encoding: [0x50'A',0xe0]
; CHECK:                              ;   fixup A - offset: 0, value: SYMBOL+3, kind: fixup_ldi

; CHECK-INST: ldi r16, 241
; CHECK-INST: ldi r29, 190
; CHECK-INST: ldi r22, 172
; CHECK-INST: ldi r27, 92
; CHECK-INST: ldi r21, 0
