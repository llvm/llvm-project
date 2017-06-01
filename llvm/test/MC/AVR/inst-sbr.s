; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:
  sbr r17, 208
  sbr r24, 190
  sbr r20, 173
  sbr r31, 0

  sbr r19, _start

; CHECK: sbr r17, 208                  ; encoding: [0x10,0x6d]
; CHECK: sbr r24, 190                  ; encoding: [0x8e,0x6b]
; CHECK: sbr r20, 173                  ; encoding: [0x4d,0x6a]
; CHECK: sbr r31, 0                    ; encoding: [0xf0,0x60]

; CHECK: sbr     r19, _start           ; encoding: [0x30'A',0x60]
; CHECK:                               ;   fixup A - offset: 0, value: _start, kind: fixup_ldi
