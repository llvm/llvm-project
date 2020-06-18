; RUN: llvm-mc -triple avr -mattr=addsubiw -show-encoding < %s | FileCheck %s
; RUNx: llvm-mc -filetype=obj -triple avr -mattr=addsubiw < %s | llvm-objdump -d --mattr=addsubiw - | FileCheck --check-prefix=CHECK-INST %s


foo:

  sbiw r26, 54
  sbiw X,   63

  sbiw 28,  52
  sbiw r28, 0

  sbiw r30, 63
  sbiw Z,   47

  sbiw r24, 1
  sbiw r24, 2

  sbiw r24, SYMBOL-1

; CHECK: sbiw r26,  54                 ; encoding: [0xd6,0x97]
; CHECK: sbiw r26,  63                 ; encoding: [0xdf,0x97]

; CHECK: sbiw r28,  52                 ; encoding: [0xe4,0x97]
; CHECK: sbiw r28,  0                  ; encoding: [0x20,0x97]

; CHECK: sbiw r30,  63                 ; encoding: [0xff,0x97]
; CHECK: sbiw r30,  47                 ; encoding: [0xbf,0x97]

; CHECK: sbiw r24,  1                  ; encoding: [0x01,0x97]
; CHECK: sbiw r24,  2                  ; encoding: [0x02,0x97]

; CHECK: sbiw    r24, SYMBOL-1         ; encoding: [0b00AAAAAA,0x97]
                                       ;   fixup A - offset: 0, value: SYMBOL-1, kind: fixup_6_adiw

; CHECK-INST: sbiw r26, 54
; CHECK-INST: sbiw X,   63

; CHECK-INST: sbiw 28,  52
; CHECK-INST: sbiw r28, 0

; CHECK-INST: sbiw r30, 63
; CHECK-INST: sbiw Z,   47

; CHECK-INST: sbiw r24, 1
; CHECK-INST: sbiw r24, 2

; CHECK-INST: sbiw r24, SYMBOL-1
