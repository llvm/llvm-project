; RUN: llvm-mc -triple avr -mattr=ijmpcall -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=ijmpcall < %s | llvm-objdump -d --mattr=ijmpcall - | FileCheck --check-prefix=CHECK-INST %s


foo:

  ijmp

; CHECK: ijmp                  ; encoding: [0x09,0x94]

; CHECK-INST: ijmp
