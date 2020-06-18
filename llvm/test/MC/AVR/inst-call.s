; RUN: llvm-mc -triple avr -mattr=jmpcall -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=jmpcall < %s | llvm-objdump -dr --mattr=jmpcall - | FileCheck -check-prefix=CHECK-INST %s


foo:

  call  4096
  call  -124
  call   -12
  call   0

; CHECK: call  4096                 ; encoding: [0x0e,0x94,0x00,0x08]
; CHECK: call -124                  ; encoding: [0xff,0x95,0xc2,0xff]
; CHECK: call -12                   ; encoding: [0xff,0x95,0xfa,0xff]
; CHECK: call  0                    ; encoding: [0x0e,0x94,0x00,0x00]


; CHECK-INST: call 4096
; CHECK-INST: call 8388484
; CHECK-INST: call 8388596
; CHECK-INST: call 0
