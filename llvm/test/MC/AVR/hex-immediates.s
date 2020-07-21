; RUN: llvm-mc -filetype=obj -triple=avr %s -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=DEC
; RUN: llvm-objdump -d --print-imm-hex %t | FileCheck %s --check-prefix=HEX

; DEC: ldi r24, 66
; HEX: ldi r24, 0x42
  ldi r24, 0x42
