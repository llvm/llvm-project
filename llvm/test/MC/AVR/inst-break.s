; RUN: llvm-mc -triple avr -mattr=break -show-encoding < %s | FileCheck %s


foo:

  break

; CHECK: break                  ; encoding: [0x98,0x95]
