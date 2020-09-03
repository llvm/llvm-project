; RUN: llvm-mc -triple avr -mattr=rmw -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=rmw < %s | llvm-objdump -d --mattr=rmw - | FileCheck -check-prefix=CHECK-INST %s


foo:

  lat Z, r13
  lat Z, r0
  lat Z, r31
  lat Z, r3

; CHECK: lat Z, r13                  ; encoding: [0xd7,0x92]
; CHECK: lat Z, r0                   ; encoding: [0x07,0x92]
; CHECK: lat Z, r31                  ; encoding: [0xf7,0x93]
; CHECK: lat Z, r3                   ; encoding: [0x37,0x92]

; CHECK-INST: lat Z, r13
; CHECK-INST: lat Z, r0
; CHECK-INST: lat Z, r31
; CHECK-INST: lat Z, r3
