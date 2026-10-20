; RUN: llvm-mc -triple avr -mattr=eijmpcall -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=eijmpcall < %s | llvm-objdump -dr --mattr=eijmpcall - | FileCheck --check-prefix=CHECK-INST %s

foo:
  eicall

; CHECK: eicall                  ; encoding: [0x19,0x95]

; CHECK-INST: eicall
