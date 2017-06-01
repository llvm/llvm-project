; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s


foo:
  ; Normal

  st X,  r10
  st X,  r17

  st Y,  r30
  st Y,  r19

  st Z,  r10
  st Z,  r2

  ; Postincrement

  st X+,  r10
  st X+,  r17

  st Y+,  r30
  st Y+,  r19

  st Z+,  r10
  st Z+,  r2

  ; Predecrement

  st -X,  r10
  st -X,  r17

  st -Y,  r30
  st -Y,  r19

  st -Z,  r10
  st -Z,  r2

; Normal

; CHECK: st X,   r10                  ; encoding: [0xac,0x92]
; CHECK: st X,   r17                  ; encoding: [0x1c,0x93]

; CHECK: st Y,   r30                  ; encoding: [0xe8,0x83]
; CHECK: st Y,   r19                  ; encoding: [0x38,0x83]

; CHECK: st Z,   r10                  ; encoding: [0xa0,0x82]
; CHECK: st Z,   r2                   ; encoding: [0x20,0x82]


; Postincrement

; CHECK: st X+,  r10                  ; encoding: [0xad,0x92]
; CHECK: st X+,  r17                  ; encoding: [0x1d,0x93]

; CHECK: st Y+,  r30                  ; encoding: [0xe9,0x93]
; CHECK: st Y+,  r19                  ; encoding: [0x39,0x93]

; CHECK: st Z+,  r10                  ; encoding: [0xa1,0x92]
; CHECK: st Z+,  r2                   ; encoding: [0x21,0x92]


; Predecrement

; CHECK: st -X,  r10                  ; encoding: [0xae,0x92]
; CHECK: st -X,  r17                  ; encoding: [0x1e,0x93]

; CHECK: st -Y,  r30                  ; encoding: [0xea,0x93]
; CHECK: st -Y,  r19                  ; encoding: [0x3a,0x93]

; CHECK: st -Z,  r10                  ; encoding: [0xa2,0x92]
; CHECK: st -Z,  r2                   ; encoding: [0x22,0x92]
