; RUN: llvm-mc -triple avr -mattr=ijmpcall -show-encoding < %s | FileCheck %s


foo:

  ijmp

; CHECK: ijmp                  ; encoding: [0x09,0x94]
