; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  cpi r16, 241
  cpi r29, 190
  cpi r22, 172
  cpi r27, 92

  cpi r21, ear

; CHECK: cpi r16, 241                  ; encoding: [0x01,0x3f]
; CHECK: cpi r29, 190                  ; encoding: [0xde,0x3b]
; CHECK: cpi r22, 172                  ; encoding: [0x6c,0x3a]
; CHECK: cpi r27, 92                   ; encoding: [0xbc,0x35]

; CHECK: cpi r21, ear                  ; encoding: [0x50'A',0x30]
; CHECK:                               ;   fixup A - offset: 0, value: ear, kind: fixup_ldi

; CHECK-INST: cpi r16, 241
; CHECK-INST: cpi r29, 190
; CHECK-INST: cpi r22, 172
; CHECK-INST: cpi r27, 92
; CHECK-INST: cpi r21, 0
