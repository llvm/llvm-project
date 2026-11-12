; RUN: llvm-mc -triple avr -mattr=rmw -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=rmw < %s | llvm-objdump -dr --mattr=rmw - | FileCheck -check-prefix=CHECK-INST %s

foo:
  lac Z, r13
  lac Z, r0
  lac Z, r31
  lac Z, r3

; CHECK: lac Z, r13                  ; encoding: [0xd6,0x92]
; CHECK: lac Z, r0                   ; encoding: [0x06,0x92]
; CHECK: lac Z, r31                  ; encoding: [0xf6,0x93]
; CHECK: lac Z, r3                   ; encoding: [0x36,0x92]

; CHECK-INST: lac Z, r13
; CHECK-INST: lac Z, r0
; CHECK-INST: lac Z, r31
; CHECK-INST: lac Z, r3
