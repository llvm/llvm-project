; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  nop

; CHECK: nop                  ; encoding: [0x00,0x00]
