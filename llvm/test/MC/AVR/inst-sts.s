; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s


foo:

  sts 3,   r5
  sts 255, r7
  sts SYMBOL+1, r25

; CHECK:  sts 3,   r5                 ; encoding: [0x50,0x92,0x03,0x00]
; CHECK:  sts 255, r7                 ; encoding: [0x70,0x92,0xff,0x00]
; CHECK:  sts SYMBOL+1, r25           ; encoding: [0x90'A',0x93'A',0x00,0x00]
; CHECK:                              ;   fixup A - offset: 0, value: SYMBOL+1, kind: fixup_16

