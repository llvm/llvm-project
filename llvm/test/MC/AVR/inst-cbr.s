; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  cbr r17, 208
  cbr r24, 190
  cbr r20, 173
  cbr r31, 0

; CHECK: andi r17, -209              ; encoding: [0x1f,0x72]
; CHECK: andi r24, -191              ; encoding: [0x81,0x74]
; CHECK: andi r20, -174              ; encoding: [0x42,0x75]
; CHECK: andi r31, -1                ; encoding: [0xff,0x7f]

; CHECK-INST: andi r17, 47
; CHECK-INST: andi r24, 65
; CHECK-INST: andi r20, 82
; CHECK-INST: andi r31, 255
