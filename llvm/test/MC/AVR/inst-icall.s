; RUN: llvm-mc -triple avr -mattr=ijmpcall -show-encoding < %s | FileCheck %s


foo:

  icall

; CHECK: icall                  ; encoding: [0x09,0x95]
