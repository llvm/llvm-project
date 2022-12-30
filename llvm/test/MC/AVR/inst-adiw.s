; RUN: llvm-mc -triple avr -mattr=addsubiw -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=addsubiw < %s | llvm-objdump --no-print-imm-hex -dr --mattr=addsubiw - | FileCheck --check-prefix=CHECK-INST %s


foo:

  adiw r26,  12
  adiw r26,  63

  adiw r28,  17
  adiw r28,  0

  adiw r30,  3
  adiw r30,  63
  adiw r24,  63
  adiw r24,  0
  adiw r30,  0

  adiw r24, SYMBOL
  adiw r24, r25

; CHECK: adiw r26,  12               ; encoding: [0x1c,0x96]
; CHECK: adiw r26,  63               ; encoding: [0xdf,0x96]

; CHECK: adiw r28,  17               ; encoding: [0x61,0x96]
; CHECK: adiw r28,  0                ; encoding: [0x20,0x96]

; CHECK: adiw r30,  3                ; encoding: [0x33,0x96]
; CHECK: adiw r30,  63               ; encoding: [0xff,0x96]
; CHECK: adiw r24,  63               ; encoding: [0xcf,0x96]
; CHECK: adiw r24,  0                ; encoding: [0x00,0x96]
; CHECK: adiw r30,  0                ; encoding: [0x30,0x96]

; CHECK: adiw r24, SYMBOL            ; encoding: [0b00AAAAAA,0x96]
                                     ;   fixup A - offset: 0, value: SYMBOL, kind: fixup_6_adiw
; CHECK: adiw r24, r25               ; encoding: [0b00AAAAAA,0x96]
                                     ;   fixup A - offset: 0, value: r25, kind: fixup_6_adiw

; CHECK-INST: adiw r26,  12
; CHECK-INST: adiw r26,  63

; CHECK-INST: adiw r28,  17
; CHECK-INST: adiw r28,  0

; CHECK-INST: adiw r30,  3
; CHECK-INST: adiw r30,  63
; CHECK-INST: adiw r24,  63
; CHECK-INST: adiw r24,  0
; CHECK-INST: adiw r30,  0

; CHECK-INST: adiw r24, 0
; CHECK-INST:      R_AVR_6_ADIW SYMBOL
; CHECK-INST: adiw r24, 0
; CHECK-INST:      R_AVR_6_ADIW r25
