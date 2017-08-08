; RUN: llvm-mc -triple avr -mattr=eijmpcall -show-encoding < %s | FileCheck %s


foo:

  eijmp

; CHECK: eijmp                  ; encoding: [0x19,0x94]
