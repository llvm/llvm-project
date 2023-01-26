; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump --no-print-imm-hex -d - | FileCheck --check-prefix=INST %s


foo:

  sbrc r2, 3
  sbrc r0, 7

; CHECK: sbrc r2, 3                  ; encoding: [0x23,0xfc]
; CHECK: sbrc r0, 7                  ; encoding: [0x07,0xfc]

; INST: sbrc r2, 3
; INST: sbrc r0, 7
