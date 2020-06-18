; RUN: llvm-mc -triple avr -mattr=jmpcall -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=jmpcall < %s | llvm-objdump -dr --mattr=jmpcall - | FileCheck -check-prefix=CHECK-INST %s


foo:

  jmp   200
  jmp  -12
  jmp   80
  jmp   0

  jmp foo+1

  jmp   0x03fffe ; Inst{16-0}  or k{16-0}
  jmp   0x7c0000 ; Inst{24-20} or k{21-17}
  jmp   0x7ffffe ; all bits set

; CHECK: jmp  200                  ; encoding: [0x0c,0x94,0x64,0x00]
; CHECK: jmp -12                   ; encoding: [0xfd,0x95,0xfa,0xff]
; CHECK: jmp  80                   ; encoding: [0x0c,0x94,0x28,0x00]
; CHECK: jmp  0                    ; encoding: [0x0c,0x94,0x00,0x00]

; CHECK: jmp foo+1                 ; encoding: [0x0c'A',0x94'A',0b00AAAAAA,0x00]
; CHECK:                           ;   fixup A - offset: 0, value: foo+1, kind: fixup_call

; CHECK: jmp 262142                ; encoding: [0x0d,0x94,0xff,0xff]
; CHECK: jmp 8126464               ; encoding: [0xfc,0x95,0x00,0x00]
; CHECK: jmp 8388606               ; encoding: [0xfd,0x95,0xff,0xff]


; CHECK-INST: jmp 200
; CHECK-INST: jmp 8388596
; CHECK-INST: jmp 80
; CHECK-INST: jmp 0
; CHECK-INST: jmp 0
; CHECK-INST:     R_AVR_CALL .text+0x1
; CHECK-INST: jmp 262142
; CHECK-INST: jmp 8126464
; CHECK-INST: jmp 8388606
