; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=sram < %s \
; RUN:     | llvm-objdump -d --mattr=sram - | FileCheck --check-prefix=INST %s


foo:

  ; Normal

  ld r10, X
  ld r17, X

  ld r30, Y
  ld r19, Y

  ld r10, Z
  ld r2,  Z

  ; Postincremenet

  ld r10, X+
  ld r17, X+

  ld r30, Y+
  ld r19, Y+

  ld r10, Z+
  ld r2,  Z+

  ; Predecrement

  ld r10, -X
  ld r17, -X

  ld r30, -Y
  ld r19, -Y

  ld r10, -Z
  ld r2,  -Z


; Normal

; CHECK: ld r10,  X                 ; encoding: [0xac,0x90]
; CHECK: ld r17,  X                 ; encoding: [0x1c,0x91]

; CHECK: ld r30,  Y                 ; encoding: [0xe8,0x81]
; CHECK: ld r19,  Y                 ; encoding: [0x38,0x81]

; CHECK: ld r10,  Z                 ; encoding: [0xa0,0x80]
; CHECK: ld r2,   Z                 ; encoding: [0x20,0x80]


; Postincrement

; CHECK: ld r10,  X+                ; encoding: [0xad,0x90]
; CHECK: ld r17,  X+                ; encoding: [0x1d,0x91]

; CHECK: ld r30,  Y+                ; encoding: [0xe9,0x91]
; CHECK: ld r19,  Y+                ; encoding: [0x39,0x91]

; CHECK: ld r10,  Z+                ; encoding: [0xa1,0x90]
; CHECK: ld r2,   Z+                ; encoding: [0x21,0x90]


; Predecrement

; CHECK: ld r10, -X                 ; encoding: [0xae,0x90]
; CHECK: ld r17, -X                 ; encoding: [0x1e,0x91]

; CHECK: ld r30, -Y                 ; encoding: [0xea,0x91]
; CHECK: ld r19, -Y                 ; encoding: [0x3a,0x91]

; CHECK: ld r10, -Z                 ; encoding: [0xa2,0x90]
; CHECK: ld r2,  -Z                 ; encoding: [0x22,0x90]

; INST: ld r10, X
; INST: ld r17, X
; INST: ldd r30, Y+0
; INST: ldd r19, Y+0
; INST: ldd r10, Z+0
; INST: ldd r2,  Z+0

; INST: ld r10, X+
; INST: ld r17, X+
; INST: ld r30, Y+
; INST: ld r19, Y+
; INST: ld r10, Z+
; INST: ld r2,  Z+

; INST: ld r10, -X
; INST: ld r17, -X
; INST: ld r30, -Y
; INST: ld r19, -Y
; INST: ld r10, -Z
; INST: ld r2,  -Z
