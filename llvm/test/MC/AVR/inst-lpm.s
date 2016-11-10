; RUN: llvm-mc -triple avr -mattr=lpm,lpmx -show-encoding < %s | FileCheck %s


foo:

  lpm

  lpm r3,  Z
  lpm r23, Z

  lpm r8, Z+
  lpm r0, Z+

; CHECK: lpm                  ; encoding: [0xc8,0x95]

; CHECK: lpm r3,  Z           ; encoding: [0x34,0x90]
; CHECK: lpm r23, Z           ; encoding: [0x74,0x91]

; CHECK: lpm r8, Z+           ; encoding: [0x85,0x90]
; CHECK: lpm r0, Z+           ; encoding: [0x05,0x90]
