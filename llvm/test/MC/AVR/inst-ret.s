; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  ret

; CHECK: ret                  ; encoding: [0x08,0x95]
