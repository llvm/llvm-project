; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  nop

; CHECK: nop                  ; encoding: [0x00,0x00]

; CHECK-INST: nop
