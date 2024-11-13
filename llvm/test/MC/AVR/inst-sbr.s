; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump --no-print-imm-hex -dr - | FileCheck --check-prefix=CHECK-INST %s

foo:
  sbr r17, 208
  sbr r24, 190
  sbr r20, 173
  sbr r31, 0

  sbr r19, _start

; CHECK: ori r17, 208                  ; encoding: [0x10,0x6d]
; CHECK: ori r24, 190                  ; encoding: [0x8e,0x6b]
; CHECK: ori r20, 173                  ; encoding: [0x4d,0x6a]
; CHECK: ori r31, 0                    ; encoding: [0xf0,0x60]

; CHECK: ori     r19, _start           ; encoding: [0x30'A',0x60]
; CHECK:                               ;   fixup A - offset: 0, value: _start, kind: fixup_ldi

; CHECK-INST: ori r17, 208
; CHECK-INST: ori r24, 190
; CHECK-INST: ori r20, 173
; CHECK-INST: ori r31, 0

; CHECK-INST: ori r19, 0
