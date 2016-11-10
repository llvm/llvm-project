; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  ori r17, 208
  ori r24, 190
  ori r20, 173
  ori r31, 0

  ori r16, FOOBAR

; CHECK: ori r17, 208           ; encoding: [0x10,0x6d]
; CHECK: ori r24, 190           ; encoding: [0x8e,0x6b]
; CHECK: ori r20, 173           ; encoding: [0x4d,0x6a]
; CHECK: ori r31, 0             ; encoding: [0xf0,0x60]

; CHECK: ori r16, FOOBAR        ; encoding: [A,0x60]
; CHECK:                        ;   fixup A - offset: 0, value: FOOBAR, kind: fixup_ldi
