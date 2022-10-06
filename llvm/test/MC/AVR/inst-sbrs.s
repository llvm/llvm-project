; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INST %s


foo:

  sbrs r2, 3
  sbrs r0, 7

; CHECK: sbrs r2, 3                  ; encoding: [0x23,0xfe]
; CHECK: sbrs r0, 7                  ; encoding: [0x07,0xfe]

; INST: sbrs r2, 3
; INST: sbrs r0, 7
