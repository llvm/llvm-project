; RUN: llvm-mc -triple avr -mattr=break -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=break < %s | llvm-objdump -dr --mattr=break - | FileCheck --check-prefix=CHECK-INST %s

foo:
  break

; CHECK: break                  ; encoding: [0x98,0x95]

; CHECK-INST: break
