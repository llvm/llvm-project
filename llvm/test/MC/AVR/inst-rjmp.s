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

; CHECK: rjmp (.Ltmp0+2)+2    ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp0+2)+2, kind: fixup_13_pcrel
; CHECK: rjmp (.Ltmp1-2)+2    ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp1-2)+2, kind: fixup_13_pcrel
; CHECK: rjmp foo             ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: foo, kind: fixup_13_pcrel
; CHECK: rjmp (.Ltmp2+8)+2    ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp2+8)+2, kind: fixup_13_pcrel
; CHECK: rjmp end             ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: end, kind: fixup_13_pcrel
; CHECK: rjmp (.Ltmp3+0)+2    ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp3+0)+2, kind: fixup_13_pcrel
; CHECK: rjmp (.Ltmp4-4)+2    ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp4-4)+2, kind: fixup_13_pcrel
; CHECK: rjmp (.Ltmp5-6)+2    ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp5-6)+2, kind: fixup_13_pcrel
; CHECK: rjmp x               ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: x, kind: fixup_13_pcrel
; CHECK: rjmp (.Ltmp6+4094)+2 ; encoding: [A,0b1100AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp6+4094)+2, kind: fixup_13_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 01 c0      rjmp  .+2
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: fd cf      rjmp  .-6
; INST-NEXT: 04 c0      rjmp  .+8
; INST-NEXT: 01 c0      rjmp  .+2
; INST-NEXT: 00 c0      rjmp  .+0
; INST-EMPTY:
; INST-LABEL: <end>:
; INST-NEXT: fe cf      rjmp  .-4
; INST-NEXT: fd cf      rjmp  .-6
; INST-EMPTY:
; INST-LABEL: <x>:
; INST-NEXT: ff cf      rjmp  .-2
; INST-NEXT: 0f c0      rjmp  .+30
; INST-NEXT: ff c7      rjmp  .+4094
