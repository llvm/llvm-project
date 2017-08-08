; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  out 4,  r2
  out 6,  r9
  out 32, r5
  out 0,  r0

  out bar-8, r29

; CHECK: out 4,  r2                  ; encoding: [0x24,0xb8]
; CHECK: out 6,  r9                  ; encoding: [0x96,0xb8]
; CHECK: out 32, r5                  ; encoding: [0x50,0xbc]
; CHECK: out 0,  r0                  ; encoding: [0x00,0xb8]

; CHECK: out bar-8, r29              ; encoding: [0xd0'A',0xb9'A']
; CHECK:                             ;   fixup A - offset: 0, value: bar-8, kind: fixup_port6

