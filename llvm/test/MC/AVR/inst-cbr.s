; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  cbr r17, 208
  cbr r24, 190
  cbr r20, 173
  cbr r31, 0

; CHECK: andi r17, -209              ; encoding: [0x1f,0x72]
; CHECK: andi r24, -191              ; encoding: [0x81,0x74]
; CHECK: andi r20, -174              ; encoding: [0x42,0x75]
; CHECK: andi r31, -1                ; encoding: [0xff,0x7f]
