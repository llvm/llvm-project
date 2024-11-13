; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -dr - | FileCheck -check-prefix=CHECK-INST %s

foo:
  dec r26
  dec r3
  dec r24
  dec r20

; CHECK: dec r26                  ; encoding: [0xaa,0x95]
; CHECK: dec r3                   ; encoding: [0x3a,0x94]
; CHECK: dec r24                  ; encoding: [0x8a,0x95]
; CHECK: dec r20                  ; encoding: [0x4a,0x95]

; CHECK-INST: dec r26
; CHECK-INST: dec r3
; CHECK-INST: dec r24
; CHECK-INST: dec r20
