; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s


foo:

  std Y+2, r2
  std Y+0, r0

  std Z+12, r9
  std Z+30, r7

  std Y+foo, r9

; CHECK: std Y+2,  r2                 ; encoding: [0x2a,0x82]
; CHECK: std Y+0,  r0                 ; encoding: [0x08,0x82]

; CHECK: std Z+12, r9                 ; encoding: [0x94,0x86]
; CHECK: std Z+30, r7                 ; encoding: [0x76,0x8e]

; CHECK: std Y+foo, r9                ; encoding: [0x98'A',0x82'A']
; CHECK:                              ;   fixup A - offset: 0, value: +foo, kind: fixup_6

