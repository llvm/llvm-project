; RUN: llvm-mc -triple avr -mattr=sram,tinyencoding -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=sram,tinyencoding < %s | llvm-objdump --no-print-imm-hex -dr --mattr=sram,tinyencoding - | FileCheck -check-prefix=CHECK-INST %s

foo:
  lds r16, 113
  lds r29, 62
  lds r22, 44
  lds r27, 92
  lds r20, SYMBOL+12
  lds r20, r21
  lds r20, z+6

; CHECK:  lds r16, 113                 ; encoding: [0x01,0xa7]
; CHECK:  lds r29, 62                  ; encoding: [0xde,0xa3]
; CHECK:  lds r22, 44                  ; encoding: [0x6c,0xa2]
; CHECK:  lds r27, 92                  ; encoding: [0xbc,0xa5]
; CHECK:  lds r20, SYMBOL+12           ; encoding: [0x40'A',0xa0'A']
; CHECK:                               ; fixup A - offset: 0, value: SYMBOL+12, kind: fixup_lds_sts_16
; CHECK:  lds r20, r21                 ; encoding: [0x40'A',0xa0'A']
; CHECK:                               ; fixup A - offset: 0, value: r21, kind: fixup_lds_sts_16
; CHECK:  lds r20, z+6                 ; encoding: [0x40'A',0xa0'A']
; CHECK:                               ; fixup A - offset: 0, value: z+6, kind: fixup_lds_sts_16

; CHECK-INST: lds r16, 113
; CHECK-INST: lds r29, 62
; CHECK-INST: lds r22, 44
; CHECK-INST: lds r27, 92
; CHECK-INST: lds r20, 0
; CHECK-INST:          R_AVR_LDS_STS_16 SYMBOL+0xc
; CHECK-INST: lds r20, 0
; CHECK-INST:          R_AVR_LDS_STS_16 r21
; CHECK-INST: lds r20, 0
; CHECK-INST:          R_AVR_LDS_STS_16 z+0x6
