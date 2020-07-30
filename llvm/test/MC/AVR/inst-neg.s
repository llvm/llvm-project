; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck -check-prefix=CHECK-INST %s

foo:
  neg r15
  neg r1
  neg r22
  neg r31

; CHECK: neg r15                  ; encoding: [0xf1,0x94]
; CHECK: neg r1                   ; encoding: [0x11,0x94]
; CHECK: neg r22                  ; encoding: [0x61,0x95]
; CHECK: neg r31                  ; encoding: [0xf1,0x95]

; CHECK-INST: neg r15
; CHECK-INST: neg r1
; CHECK-INST: neg r22
; CHECK-INST: neg r31
