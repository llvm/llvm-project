; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=sram < %s \
; RUN:     | llvm-objdump -d --mattr=sram - | FileCheck --check-prefix=INST %s

foo:
  ldd r2, Y+2
  ldd r0, Y+0

  ldd r9, Z+12
  ldd r7, Z+30

  ldd r9, Z+foo

; CHECK: ldd r2, Y+2                  ; encoding: [0x2a,0x80]
; CHECK: ldd r0, Y+0                  ; encoding: [0x08,0x80]

; CHECK: ldd r9, Z+12                 ; encoding: [0x94,0x84]
; CHECK: ldd r7, Z+30                 ; encoding: [0x76,0x8c]

; CHECK: ldd r9, Z+foo                ; encoding: [0x90'A',0x80'A']
; CHECK:                              ;   fixup A - offset: 0, value: +foo, kind: fixup_6

; INST: ldd r2, Y+2
; INST: ldd r0, Y+0
; INST: ldd r9, Z+12
; INST: ldd r7, Z+30
