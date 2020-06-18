; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  sbci r17, 21
  sbci r23, 196
  sbci r30, 244
  sbci r19, 16
  sbci r22, FOO

; CHECK: sbci r17, 21                 ; encoding: [0x15,0x41]
; CHECK: sbci r23, 196                ; encoding: [0x74,0x4c]
; CHECK: sbci r30, 244                ; encoding: [0xe4,0x4f]
; CHECK: sbci r19, 16                 ; encoding: [0x30,0x41]

; CHECK: sbci r22, FOO                ; encoding: [0x60'A',0x40]
; CHECK:                              ;   fixup A - offset: 0, value: FOO, kind: fixup_ldi

; CHECK-INST: sbci r17, 21
; CHECK-INST: sbci r23, 196
; CHECK-INST: sbci r30, 244
; CHECK-INST: sbci r19, 16
; CHECK-INST: sbci r22, 0
