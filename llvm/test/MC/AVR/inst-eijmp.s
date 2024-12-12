; RUN: llvm-mc -triple avr -mattr=eijmpcall -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=eijmpcall < %s | llvm-objdump -dr --mattr=eijmpcall - | FileCheck --check-prefix=CHECK-INST %s

foo:
  eijmp

; CHECK: eijmp                  ; encoding: [0x19,0x94]

; CHECK-INST: eijmp
