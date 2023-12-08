; RUN: llvm-mc -triple avr -mattr=sram,tinyencoding -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=sram,tinyencoding < %s | llvm-objdump --no-print-imm-hex -dr --mattr=sram,tinyencoding - | FileCheck -check-prefix=CHECK-INST %s

foo:
  sts 3,        r16
  sts 127,      r17
  sts SYMBOL+1, r25
  sts x,        r25
  sts r25+1,    r25

; CHECK:  sts 3,        r16           ; encoding: [0x03,0xa8]
; CHECK:  sts 127,      r17           ; encoding: [0x1f,0xaf]
; CHECK:  sts SYMBOL+1, r25           ; encoding: [0x90'A',0xa8'A']
; CHECK:                              ; fixup A - offset: 0, value: SYMBOL+1, kind: fixup_lds_sts_16
; CHECK:  sts x,        r25           ; encoding: [0x90'A',0xa8'A']
; CHECK:                              ; fixup A - offset: 0, value: x, kind: fixup_lds_sts_16
; CHECK:  sts r25+1,    r25           ; encoding: [0x90'A',0xa8'A']
; CHECK:                              ; fixup A - offset: 0, value: r25+1, kind: fixup_lds_sts_16

; CHECK-INST: sts 3,   r16
; CHECK-INST: sts 127, r17
; CHECK-INST: sts 0,   r25
; CHECK-INST:     R_AVR_LDS_STS_16 SYMBOL+0x1
; CHECK-INST: sts 0,   r25
; CHECK-INST:     R_AVR_LDS_STS_16 x
; CHECK-INST: sts 0,   r25
; CHECK-INST:     R_AVR_LDS_STS_16 r25+0x1
