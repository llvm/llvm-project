; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  asr r31
  asr r25
  asr r5
  asr r0

; CHECK: asr r31                ; encoding: [0xf5,0x95]
; CHECK: asr r25                ; encoding: [0x95,0x95]
; CHECK: asr r5                 ; encoding: [0x55,0x94]
; CHECK: asr r0                 ; encoding: [0x05,0x94]
