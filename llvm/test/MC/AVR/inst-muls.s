; RUN: llvm-mc -triple avr -mattr=mul -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=mul < %s | llvm-objdump -dr --mattr=mul - | FileCheck -check-prefix=CHECK-INST %s

foo:
  muls r22, r16
  muls r19, r17
  muls r28, r31
  muls r31, r31
  muls r16, r16
  muls r16, r31

; CHECK: muls r22, r16                   ; encoding: [0x60,0x02]
; CHECK: muls r19, r17                   ; encoding: [0x31,0x02]
; CHECK: muls r28, r31                   ; encoding: [0xcf,0x02]
; CHECK: muls r31, r31                   ; encoding: [0xff,0x02]
; CHECK: muls r16, r16                   ; encoding: [0x00,0x02]
; CHECK: muls r16, r31                   ; encoding: [0x0f,0x02]

; CHECK-INST: muls r22, r16
; CHECK-INST: muls r19, r17
; CHECK-INST: muls r28, r31
; CHECK-INST: muls r31, r31
; CHECK-INST: muls r16, r16
; CHECK-INST: muls r16, r31
