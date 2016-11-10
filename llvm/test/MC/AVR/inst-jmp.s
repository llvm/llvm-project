; RUN: llvm-mc -triple avr -mattr=jmpcall -show-encoding < %s | FileCheck %s


foo:

  jmp   200
  jmp  -12
  jmp   80
  jmp   0

  jmp foo+1

; CHECK: jmp  200                  ; encoding: [0x0c,0x94,0x64,0x00]
; CHECK: jmp -12                   ; encoding: [0xfd,0x95,0xfa,0xff]
; CHECK: jmp  80                   ; encoding: [0x0c,0x94,0x28,0x00]
; CHECK: jmp  0                    ; encoding: [0x0c,0x94,0x00,0x00]

; CHECK: jmp foo+1                 ; encoding: [0x0c'A',0x94'A',0b00AAAAAA,0x00]
; CHECK:                           ;   fixup A - offset: 0, value: foo+1, kind: fixup_call
