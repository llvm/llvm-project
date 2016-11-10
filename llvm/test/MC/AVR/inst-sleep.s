; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  sleep

; CHECK: sleep                  ; encoding: [0x88,0x95]
