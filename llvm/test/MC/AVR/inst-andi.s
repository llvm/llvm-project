; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump --no-print-imm-hex -dr - | FileCheck --check-prefix=CHECK-INST %s

foo:
  andi r16, 255
  andi r29, 190
  andi r22, 172
  andi r27, 92

  andi r20, BAR

; CHECK: andi r16, 255                 ; encoding: [0x0f,0x7f]
; CHECK: andi r29, 190                 ; encoding: [0xde,0x7b]
; CHECK: andi r22, 172                 ; encoding: [0x6c,0x7a]
; CHECK: andi r27, 92                  ; encoding: [0xbc,0x75]

; CHECK: andi r20, BAR                 ; encoding: [0x40'A',0x70]
; CHECK:                               ;   fixup A - offset: 0, value: BAR, kind: fixup_ldi

; CHECK-INST: andi r16, 255
; CHECK-INST: andi r29, 190
; CHECK-INST: andi r22, 172
; CHECK-INST: andi r27, 92

; CHECK-INST: andi r20, 0
