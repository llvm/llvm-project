; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INST %s


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

; CHECK: rjmp    .Ltmp0+2                ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp0+2, kind: fixup_13_pcrel
; CHECK: rjmp    .Ltmp1-2                ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp1-2, kind: fixup_13_pcrel
; CHECK: rjmp    foo                     ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: foo, kind: fixup_13_pcrel
; CHECK: rjmp    .Ltmp2+8                ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp2+8, kind: fixup_13_pcrel
; CHECK: rjmp    end                     ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: end, kind: fixup_13_pcrel
; CHECK: rjmp    .Ltmp3+0                ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp3+0, kind: fixup_13_pcrel
; CHECK: rjmp    .Ltmp4-4                ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp4-4, kind: fixup_13_pcrel
; CHECK: rjmp    .Ltmp5-6                ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp5-6, kind: fixup_13_pcrel
; CHECK: rjmp    x                       ; encoding: [A,0b1100AAAA]
; CHECK:                                 ;   fixup A - offset: 0, value: x, kind: fixup_13_pcrel

; INST: 01 c0      rjmp  .+2
; INST: ff cf      rjmp  .-2
; INST: 00 c0      rjmp  .+0
; INST: 04 c0      rjmp  .+8
; INST: 00 c0      rjmp  .+0
; INST: 00 c0      rjmp  .+0
; INST: fe cf      rjmp  .-4
; INST: fd cf      rjmp  .-6
; INST: 00 c0      rjmp  .+0
; INST: 0f c0      rjmp  .+30
