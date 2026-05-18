; RUN: llvm-mc -triple avr -mattr=addsubiw -show-encoding < %s | FileCheck %s

foo:
  sbiw r24,     1
  sbiw r25:r24, 2
  sbiw r24,     2
  sbiw r27:r26, 3

; CHECK: sbiw r24,  1                  ; encoding: [0x01,0x97]
; CHECK: sbiw r24,  2                  ; encoding: [0x02,0x97]
; CHECK: sbiw r24,  2                  ; encoding: [0x02,0x97]
; CHECK: sbiw r26,  3                  ; encoding: [0x13,0x97]
