; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:
  ser r16
  ser r31
  ser r27
  ser r31

; CHECK: ldi r16, 255              ; encoding: [0x0f,0xef]
; CHECK: ldi r31, 255              ; encoding: [0xff,0xef]
; CHECK: ldi r27, 255              ; encoding: [0xbf,0xef]
; CHECK: ldi r31, 255              ; encoding: [0xff,0xef]
