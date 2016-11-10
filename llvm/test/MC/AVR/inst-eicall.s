; RUN: llvm-mc -triple avr -mattr=eijmpcall -show-encoding < %s | FileCheck %s


foo:

  eicall

; CHECK: eicall                  ; encoding: [0x19,0x95]
