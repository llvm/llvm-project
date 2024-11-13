; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump --no-print-imm-hex -dr - | FileCheck --check-prefix=CHECK-INST %s

foo:
  subi r22, 82
  subi r27, 39
  subi r31, 244
  subi r16, 144

  subi r20, EXTERN_SYMBOL+0

; CHECK: subi r22, 82                  ; encoding: [0x62,0x55]
; CHECK: subi r27, 39                  ; encoding: [0xb7,0x52]
; CHECK: subi r31, 244                 ; encoding: [0xf4,0x5f]
; CHECK: subi r16, 144                 ; encoding: [0x00,0x59]

; CHECK: subi    r20, EXTERN_SYMBOL+0  ; encoding: [0x40'A',0x50]
; CHECK:                               ;   fixup A - offset: 0, value: EXTERN_SYMBOL+0, kind: fixup_ldi

; CHECK-INST: subi r22, 82
; CHECK-INST: subi r27, 39
; CHECK-INST: subi r31, 244
; CHECK-INST: subi r16, 144

; CHECK-INST: subi r20, 0
