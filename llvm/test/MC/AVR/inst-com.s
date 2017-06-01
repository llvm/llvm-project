; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  com r30
  com r17
  com r4
  com r0

; CHECK: com r30                  ; encoding: [0xe0,0x95]
; CHECK: com r17                  ; encoding: [0x10,0x95]
; CHECK: com r4                   ; encoding: [0x40,0x94]
; CHECK: com r0                   ; encoding: [0x00,0x94]
