; RUN: llvm-mc -triple avr -mattr=rmw -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=rmw < %s | llvm-objdump -d --mattr=rmw - | FileCheck -check-prefix=CHECK-INST %s


foo:

  las Z, r13
  las Z, r0
  las Z, r31
  las Z, r3

; CHECK: las Z, r13                  ; encoding: [0xd5,0x92]
; CHECK: las Z, r0                   ; encoding: [0x05,0x92]
; CHECK: las Z, r31                  ; encoding: [0xf5,0x93]
; CHECK: las Z, r3                   ; encoding: [0x35,0x92]

; CHECK-INST: las Z, r13
; CHECK-INST: las Z, r0
; CHECK-INST: las Z, r31
; CHECK-INST: las Z, r3
