; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -dr - | FileCheck --check-prefix=CHECK-INST %s

foo:
  wdr

; CHECK: wdr                  ; encoding: [0xa8,0x95]

; CHECK-INST: wdr
