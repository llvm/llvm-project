; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  reti

; CHECK: reti                  ; encoding: [0x18,0x95]
