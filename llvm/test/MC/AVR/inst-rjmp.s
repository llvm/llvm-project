; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  rjmp  .+2
  rjmp  .-2
  rjmp  foo
  rjmp  .+8
  rjmp  end
  rjmp  .+0

end:
  rjmp .-4
  rjmp .-6

x:
  rjmp x
  .short 0xc00f
  rjmp .+4094

; CHECK: rjmp .Ltmp0+2+2    ; encoding: [A,0b1100AAAA]
; CHECK: rjmp .Ltmp1-2+2    ; encoding: [A,0b1100AAAA]
; CHECK: rjmp foo             ; encoding: [A,0b1100AAAA]
; CHECK: rjmp .Ltmp2+8+2    ; encoding: [A,0b1100AAAA]
; CHECK: rjmp end             ; encoding: [A,0b1100AAAA]
; CHECK: rjmp .Ltmp3+0+2    ; encoding: [A,0b1100AAAA]
; CHECK: rjmp .Ltmp4-4+2    ; encoding: [A,0b1100AAAA]
; CHECK: rjmp .Ltmp5-6+2    ; encoding: [A,0b1100AAAA]
; CHECK: rjmp x               ; encoding: [A,0b1100AAAA]
; CHECK: rjmp .Ltmp6+4094+2 ; encoding: [A,0b1100AAAA]

; INST-LABEL: <foo>:
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0x4
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0x2
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0x10
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0xc
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0xc
; INST-EMPTY:
; INST-LABEL: <end>:
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0xa
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0xa
; INST-EMPTY:
; INST-LABEL: <x>:
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0x10
; INST-NEXT: 0f c0      rjmp  .+30
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: R_AVR_13_PCREL .text+0x1014
