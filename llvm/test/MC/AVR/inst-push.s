; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=sram < %s | llvm-objdump -d --mattr=sram - | FileCheck -check-prefix=CHECK-INST %s


foo:

  push r31
  push r25
  push r5
  push r0

; CHECK: push r31                ; encoding: [0xff,0x93]
; CHECK: push r25                ; encoding: [0x9f,0x93]
; CHECK: push r5                 ; encoding: [0x5f,0x92]
; CHECK: push r0                 ; encoding: [0x0f,0x92]

; CHECK-INST: push r31
; CHECK-INST: push r25
; CHECK-INST: push r5
; CHECK-INST: push r0
