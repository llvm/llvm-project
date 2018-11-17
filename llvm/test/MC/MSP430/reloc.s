; RUN: llvm-mc -triple msp430 -show-encoding < %s | FileCheck %s

         mov    disp+2(r8), r15
; CHECK: mov    disp+2(r8), r15 ; encoding: [0x1f,0x48,A,A]
; CHECK:                        ;   fixup A - offset: 2, value: disp+2, kind: fixup_16_byte

         mov    disp+2, r15
; CHECK: mov    disp+2, r15     ; encoding: [0x1f,0x40,A,A]
; CHECK:                        ;   fixup A - offset: 2, value: disp+2, kind: fixup_16_pcrel_byte

         mov    &disp+2, r15
; CHECK: mov    &disp+2, r15    ; encoding: [0x1f,0x42,A,A]
; CHECK:                        ;   fixup A - offset: 2, value: disp+2, kind: fixup_16

         mov    disp, disp+2
; CHECK: mov    disp, disp+2    ; encoding: [0x90,0x40,A,A,B,B]
; CHECK:                        ;   fixup A - offset: 2, value: disp, kind: fixup_16_pcrel_byte
; CHECK:                        ;   fixup B - offset: 4, value: disp+2, kind: fixup_16_pcrel_byte

         jmp    foo
; CHECK: jmp    foo             ; encoding: [A,0b001111AA]
; CHECK:                        ;   fixup A - offset: 0, value: foo, kind: fixup_10_pcrel

; RUN: llvm-mc -filetype=obj -triple msp430 < %s | llvm-readobj -r \
; RUN:   | FileCheck -check-prefix=RELOC %s
.short  _ctype+3
; RELOC: R_MSP430_16_BYTE _ctype 0x3
