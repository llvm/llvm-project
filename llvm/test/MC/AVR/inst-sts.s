; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=sram < %s | llvm-objdump --no-print-imm-hex -dr --mattr=sram - | FileCheck -check-prefix=CHECK-INST %s


foo:

  sts 3,        r5
  sts 255,      r7
  sts SYMBOL+1, r25
  sts r25,      r25
  sts y+3,      r25

; CHECK:  sts 3,        r5            ; encoding: [0x50,0x92,0x03,0x00]
; CHECK:  sts 255,      r7            ; encoding: [0x70,0x92,0xff,0x00]
; CHECK:  sts SYMBOL+1, r25           ; encoding: [0x90,0x93,A,A]
; CHECK:                              ;   fixup A - offset: 2, value: SYMBOL+1, kind: fixup_16
; CHECK:  sts r25,      r25           ; encoding: [0x90,0x93,A,A]
; CHECK:                              ;   fixup A - offset: 2, value: r25, kind: fixup_16
; CHECK:  sts y+3,      r25           ; encoding: [0x90,0x93,A,A]
; CHECK:                              ;   fixup A - offset: 2, value: y+3, kind: fixup_16


; CHECK-INST: sts 3,   r5
; CHECK-INST: sts 255, r7
; CHECK-INST: sts 0,   r25
; CHECK-INST:     R_AVR_16 SYMBOL+0x1
; CHECK-INST: sts 0,   r25
; CHECK-INST:     R_AVR_16 r25
; CHECK-INST: sts 0,   r25
; CHECK-INST:     R_AVR_16 y+0x3
