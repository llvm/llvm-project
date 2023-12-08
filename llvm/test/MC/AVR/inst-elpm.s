; RUN: llvm-mc -triple avr -mattr=elpm,elpmx -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=elpm,elpmx < %s | llvm-objdump -d --mattr=elpm,elpmx - | FileCheck -check-prefix=CHECK-INST %s


foo:

  elpm

  elpm r3,  Z
  elpm r23, Z

  elpm r8, Z+
  elpm r0, Z+

; CHECK: elpm                  ; encoding: [0xd8,0x95]

; CHECK: elpm r3,  Z           ; encoding: [0x36,0x90]
; CHECK: elpm r23, Z           ; encoding: [0x76,0x91]

; CHECK: elpm r8, Z+           ; encoding: [0x87,0x90]
; CHECK: elpm r0, Z+           ; encoding: [0x07,0x90]

; CHECK-INST: elpm

; CHECK-INST: elpm r3,  Z
; CHECK-INST: elpm r23, Z

; CHECK-INST: elpm r8, Z+
; CHECK-INST: elpm r0, Z+
