; RUN: llvm-mc -triple avr -mattr=rmw -show-encoding < %s | FileCheck %s


foo:

  lac Z, r13
  lac Z, r0
  lac Z, r31
  lac Z, r3

; CHECK: lac Z, r13                  ; encoding: [0xd6,0x92]
; CHECK: lac Z, r0                   ; encoding: [0x06,0x92]
; CHECK: lac Z, r31                  ; encoding: [0xf6,0x93]
; CHECK: lac Z, r3                   ; encoding: [0x36,0x92]
