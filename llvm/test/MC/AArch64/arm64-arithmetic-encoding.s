; RUN: llvm-mc -triple arm64-apple-darwin -mattr=neon -show-encoding < %s | FileCheck %s

foo:
;==---------------------------------------------------------------------------==
; Add/Subtract with carry/borrow
;==---------------------------------------------------------------------------==

  adc   w1, w2, w3
  adc   x1, x2, x3
  adcs  w5, w4, w3
  adcs  x5, x4, x3

; CHECK: adc  w1, w2, w3             ; encoding: [0x41,0x00,0x03,0x1a]
; CHECK: adc  x1, x2, x3             ; encoding: [0x41,0x00,0x03,0x9a]
; CHECK: adcs w5, w4, w3             ; encoding: [0x85,0x00,0x03,0x3a]
; CHECK: adcs x5, x4, x3             ; encoding: [0x85,0x00,0x03,0xba]

  sbc   w1, w2, w3
  sbc   x1, x2, x3
  sbcs  w1, w2, w3
  sbcs  x1, x2, x3

; CHECK: sbc  w1, w2, w3             ; encoding: [0x41,0x00,0x03,0x5a]
; CHECK: sbc  x1, x2, x3             ; encoding: [0x41,0x00,0x03,0xda]
; CHECK: sbcs w1, w2, w3             ; encoding: [0x41,0x00,0x03,0x7a]
; CHECK: sbcs x1, x2, x3             ; encoding: [0x41,0x00,0x03,0xfa]

;==---------------------------------------------------------------------------==
; Add/Subtract with (optionally shifted) immediate
;==---------------------------------------------------------------------------==

  add w3, w4, #1024
  add w3, w4, #1024, lsl #0
  add x3, x4, #1024
  add x3, x4, #1024, lsl #0

; CHECK: add w3, w4, #1024           ; encoding: [0x83,0x00,0x10,0x11]
; CHECK: add w3, w4, #1024           ; encoding: [0x83,0x00,0x10,0x11]
; CHECK: add x3, x4, #1024           ; encoding: [0x83,0x00,0x10,0x91]
; CHECK: add x3, x4, #1024           ; encoding: [0x83,0x00,0x10,0x91]

  add w3, w4, #1024, lsl #12
  add w3, w4, #4194304
  add w3, w4, #0, lsl #12
  add x3, x4, #1024, lsl #12
  add x3, x4, #4194304
  add x3, x4, #0, lsl #12
  add sp, sp, #32

; CHECK: add w3, w4, #1024, lsl #12  ; encoding: [0x83,0x00,0x50,0x11]
; CHECK: add w3, w4, #1024, lsl #12  ; encoding: [0x83,0x00,0x50,0x11]
; CHECK: add w3, w4, #0, lsl #12     ; encoding: [0x83,0x00,0x40,0x11]
; CHECK: add x3, x4, #1024, lsl #12  ; encoding: [0x83,0x00,0x50,0x91]
; CHECK: add x3, x4, #1024, lsl #12  ; encoding: [0x83,0x00,0x50,0x91]
; CHECK: add x3, x4, #0, lsl #12     ; encoding: [0x83,0x00,0x40,0x91]
; CHECK: add sp, sp, #32             ; encoding: [0xff,0x83,0x00,0x91]

  adds w3, w4, #1024
  adds w3, w4, #1024, lsl #0
  adds w3, w4, #1024, lsl #12
  adds x3, x4, #1024
  adds x3, x4, #1024, lsl #0
  adds x3, x4, #1024, lsl #12

; CHECK: adds w3, w4, #1024          ; encoding: [0x83,0x00,0x10,0x31]
; CHECK: adds w3, w4, #1024          ; encoding: [0x83,0x00,0x10,0x31]
; CHECK: adds w3, w4, #1024, lsl #12 ; encoding: [0x83,0x00,0x50,0x31]
; CHECK: adds x3, x4, #1024          ; encoding: [0x83,0x00,0x10,0xb1]
; CHECK: adds x3, x4, #1024          ; encoding: [0x83,0x00,0x10,0xb1]
; CHECK: adds x3, x4, #1024, lsl #12 ; encoding: [0x83,0x00,0x50,0xb1]

  sub w3, w4, #1024
  sub w3, w4, #1024, lsl #0
  sub w3, w4, #1024, lsl #12
  sub x3, x4, #1024
  sub x3, x4, #1024, lsl #0
  sub x3, x4, #1024, lsl #12
  sub sp, sp, #32

; CHECK: sub w3, w4, #1024           ; encoding: [0x83,0x00,0x10,0x51]
; CHECK: sub w3, w4, #1024           ; encoding: [0x83,0x00,0x10,0x51]
; CHECK: sub w3, w4, #1024, lsl #12  ; encoding: [0x83,0x00,0x50,0x51]
; CHECK: sub x3, x4, #1024           ; encoding: [0x83,0x00,0x10,0xd1]
; CHECK: sub x3, x4, #1024           ; encoding: [0x83,0x00,0x10,0xd1]
; CHECK: sub x3, x4, #1024, lsl #12  ; encoding: [0x83,0x00,0x50,0xd1]
; CHECK: sub sp, sp, #32             ; encoding: [0xff,0x83,0x00,0xd1]

  subs w3, w4, #1024
  subs w3, w4, #1024, lsl #0
  subs w3, w4, #1024, lsl #12
  subs x3, x4, #1024
  subs x3, x4, #1024, lsl #0
  subs x3, x4, #1024, lsl #12

; CHECK: subs w3, w4, #1024          ; encoding: [0x83,0x00,0x10,0x71]
; CHECK: subs w3, w4, #1024          ; encoding: [0x83,0x00,0x10,0x71]
; CHECK: subs w3, w4, #1024, lsl #12 ; encoding: [0x83,0x00,0x50,0x71]
; CHECK: subs x3, x4, #1024          ; encoding: [0x83,0x00,0x10,0xf1]
; CHECK: subs x3, x4, #1024          ; encoding: [0x83,0x00,0x10,0xf1]
; CHECK: subs x3, x4, #1024, lsl #12 ; encoding: [0x83,0x00,0x50,0xf1]

;==---------------------------------------------------------------------------==
; Add/Subtract register with (optional) shift
;==---------------------------------------------------------------------------==

  add w12, w13, w14
  add x12, x13, x14
  add w12, w13, w14, lsl #12
  add x12, x13, x14, lsl #12
  add x12, x13, x14, lsr #42
  add x12, x13, x14, asr #39

; CHECK: add w12, w13, w14           ; encoding: [0xac,0x01,0x0e,0x0b]
; CHECK: add x12, x13, x14           ; encoding: [0xac,0x01,0x0e,0x8b]
; CHECK: add w12, w13, w14, lsl #12  ; encoding: [0xac,0x31,0x0e,0x0b]
; CHECK: add x12, x13, x14, lsl #12  ; encoding: [0xac,0x31,0x0e,0x8b]
; CHECK: add x12, x13, x14, lsr #42  ; encoding: [0xac,0xa9,0x4e,0x8b]
; CHECK: add x12, x13, x14, asr #39  ; encoding: [0xac,0x9d,0x8e,0x8b]

  sub w12, w13, w14
  sub x12, x13, x14
  sub w12, w13, w14, lsl #12
  sub x12, x13, x14, lsl #12
  sub x12, x13, x14, lsr #42
  sub x12, x13, x14, asr #39

; CHECK: sub w12, w13, w14           ; encoding: [0xac,0x01,0x0e,0x4b]
; CHECK: sub x12, x13, x14           ; encoding: [0xac,0x01,0x0e,0xcb]
; CHECK: sub w12, w13, w14, lsl #12  ; encoding: [0xac,0x31,0x0e,0x4b]
; CHECK: sub x12, x13, x14, lsl #12  ; encoding: [0xac,0x31,0x0e,0xcb]
; CHECK: sub x12, x13, x14, lsr #42  ; encoding: [0xac,0xa9,0x4e,0xcb]
; CHECK: sub x12, x13, x14, asr #39  ; encoding: [0xac,0x9d,0x8e,0xcb]

  adds w12, w13, w14
  adds x12, x13, x14
  adds w12, w13, w14, lsl #12
  adds x12, x13, x14, lsl #12
  adds x12, x13, x14, lsr #42
  adds x12, x13, x14, asr #39

; CHECK: adds w12, w13, w14          ; encoding: [0xac,0x01,0x0e,0x2b]
; CHECK: adds x12, x13, x14          ; encoding: [0xac,0x01,0x0e,0xab]
; CHECK: adds w12, w13, w14, lsl #12 ; encoding: [0xac,0x31,0x0e,0x2b]
; CHECK: adds x12, x13, x14, lsl #12 ; encoding: [0xac,0x31,0x0e,0xab]
; CHECK: adds x12, x13, x14, lsr #42 ; encoding: [0xac,0xa9,0x4e,0xab]
; CHECK: adds x12, x13, x14, asr #39 ; encoding: [0xac,0x9d,0x8e,0xab]

  subs w12, w13, w14
  subs x12, x13, x14
  subs w12, w13, w14, lsl #12
  subs x12, x13, x14, lsl #12
  subs x12, x13, x14, lsr #42
  subs x12, x13, x14, asr #39

; CHECK: subs w12, w13, w14          ; encoding: [0xac,0x01,0x0e,0x6b]
; CHECK: subs x12, x13, x14          ; encoding: [0xac,0x01,0x0e,0xeb]
; CHECK: subs w12, w13, w14, lsl #12 ; encoding: [0xac,0x31,0x0e,0x6b]
; CHECK: subs x12, x13, x14, lsl #12 ; encoding: [0xac,0x31,0x0e,0xeb]
; CHECK: subs x12, x13, x14, lsr #42 ; encoding: [0xac,0xa9,0x4e,0xeb]
; CHECK: subs x12, x13, x14, asr #39 ; encoding: [0xac,0x9d,0x8e,0xeb]

; Check use of upper case register names rdar://14354073
  add X2, X2, X2
; CHECK: add x2, x2, x2              ; encoding: [0x42,0x00,0x02,0x8b]

;==---------------------------------------------------------------------------==
; Add/Subtract with (optional) extend
;==---------------------------------------------------------------------------==

  add w1, w2, w3, uxtb
  add w1, w2, w3, uxth
  add w1, w2, w3, uxtw
  add w1, w2, w3, uxtx
  add w1, w2, w3, sxtb
  add w1, w2, w3, sxth
  add w1, w2, w3, sxtw
  add w1, w2, w3, sxtx

; CHECK: add w1, w2, w3, uxtb        ; encoding: [0x41,0x00,0x23,0x0b]
; CHECK: add w1, w2, w3, uxth        ; encoding: [0x41,0x20,0x23,0x0b]
; CHECK: add w1, w2, w3, uxtw        ; encoding: [0x41,0x40,0x23,0x0b]
; CHECK: add w1, w2, w3, uxtx        ; encoding: [0x41,0x60,0x23,0x0b]
; CHECK: add w1, w2, w3, sxtb        ; encoding: [0x41,0x80,0x23,0x0b]
; CHECK: add w1, w2, w3, sxth        ; encoding: [0x41,0xa0,0x23,0x0b]
; CHECK: add w1, w2, w3, sxtw        ; encoding: [0x41,0xc0,0x23,0x0b]
; CHECK: add w1, w2, w3, sxtx        ; encoding: [0x41,0xe0,0x23,0x0b]

  add x1, x2, w3, uxtb
  add x1, x2, w3, uxth
  add x1, x2, w3, uxtw
  add x1, x2, w3, sxtb
  add x1, x2, w3, sxth
  add x1, x2, w3, sxtw

; CHECK: add x1, x2, w3, uxtb        ; encoding: [0x41,0x00,0x23,0x8b]
; CHECK: add x1, x2, w3, uxth        ; encoding: [0x41,0x20,0x23,0x8b]
; CHECK: add x1, x2, w3, uxtw        ; encoding: [0x41,0x40,0x23,0x8b]
; CHECK: add x1, x2, w3, sxtb        ; encoding: [0x41,0x80,0x23,0x8b]
; CHECK: add x1, x2, w3, sxth        ; encoding: [0x41,0xa0,0x23,0x8b]
; CHECK: add x1, x2, w3, sxtw        ; encoding: [0x41,0xc0,0x23,0x8b]

  add w1, wsp, w3
  add w1, wsp, w3, uxtw #0
  add w2, wsp, w3, lsl #1
  add sp, x2, x3
  add sp, x2, x3, uxtx #0

; CHECK: add w1, wsp, w3             ; encoding: [0xe1,0x43,0x23,0x0b]
; CHECK: add w1, wsp, w3             ; encoding: [0xe1,0x43,0x23,0x0b]
; CHECK: add w2, wsp, w3, lsl #1     ; encoding: [0xe2,0x47,0x23,0x0b]
; CHECK: add sp, x2, x3              ; encoding: [0x5f,0x60,0x23,0x8b]
; CHECK: add sp, x2, x3              ; encoding: [0x5f,0x60,0x23,0x8b]

  sub w1, w2, w3, uxtb
  sub w1, w2, w3, uxth
  sub w1, w2, w3, uxtw
  sub w1, w2, w3, uxtx
  sub w1, w2, w3, sxtb
  sub w1, w2, w3, sxth
  sub w1, w2, w3, sxtw
  sub w1, w2, w3, sxtx

; CHECK: sub w1, w2, w3, uxtb        ; encoding: [0x41,0x00,0x23,0x4b]
; CHECK: sub w1, w2, w3, uxth        ; encoding: [0x41,0x20,0x23,0x4b]
; CHECK: sub w1, w2, w3, uxtw        ; encoding: [0x41,0x40,0x23,0x4b]
; CHECK: sub w1, w2, w3, uxtx        ; encoding: [0x41,0x60,0x23,0x4b]
; CHECK: sub w1, w2, w3, sxtb        ; encoding: [0x41,0x80,0x23,0x4b]
; CHECK: sub w1, w2, w3, sxth        ; encoding: [0x41,0xa0,0x23,0x4b]
; CHECK: sub w1, w2, w3, sxtw        ; encoding: [0x41,0xc0,0x23,0x4b]
; CHECK: sub w1, w2, w3, sxtx        ; encoding: [0x41,0xe0,0x23,0x4b]

  sub x1, x2, w3, uxtb
  sub x1, x2, w3, uxth
  sub x1, x2, w3, uxtw
  sub x1, x2, w3, sxtb
  sub x1, x2, w3, sxth
  sub x1, x2, w3, sxtw

; CHECK: sub x1, x2, w3, uxtb        ; encoding: [0x41,0x00,0x23,0xcb]
; CHECK: sub x1, x2, w3, uxth        ; encoding: [0x41,0x20,0x23,0xcb]
; CHECK: sub x1, x2, w3, uxtw        ; encoding: [0x41,0x40,0x23,0xcb]
; CHECK: sub x1, x2, w3, sxtb        ; encoding: [0x41,0x80,0x23,0xcb]
; CHECK: sub x1, x2, w3, sxth        ; encoding: [0x41,0xa0,0x23,0xcb]
; CHECK: sub x1, x2, w3, sxtw        ; encoding: [0x41,0xc0,0x23,0xcb]

  sub w1, wsp, w3
  sub w1, wsp, w3, uxtw #0
  sub sp, x2, x3
  sub sp, x2, x3, uxtx #0
  sub sp, x3, x7, lsl #4

; CHECK: sub w1, wsp, w3             ; encoding: [0xe1,0x43,0x23,0x4b]
; CHECK: sub w1, wsp, w3             ; encoding: [0xe1,0x43,0x23,0x4b]
; CHECK: sub sp, x2, x3              ; encoding: [0x5f,0x60,0x23,0xcb]
; CHECK: sub sp, x2, x3              ; encoding: [0x5f,0x60,0x23,0xcb]
; CHECK: sp, x3, x7, lsl #4          ; encoding: [0x7f,0x70,0x27,0xcb]

  adds w1, w2, w3, uxtb
  adds w1, w2, w3, uxth
  adds w1, w2, w3, uxtw
  adds w1, w2, w3, uxtx
  adds w1, w2, w3, sxtb
  adds w1, w2, w3, sxth
  adds w1, w2, w3, sxtw
  adds w1, w2, w3, sxtx

; CHECK: adds w1, w2, w3, uxtb       ; encoding: [0x41,0x00,0x23,0x2b]
; CHECK: adds w1, w2, w3, uxth       ; encoding: [0x41,0x20,0x23,0x2b]
; CHECK: adds w1, w2, w3, uxtw       ; encoding: [0x41,0x40,0x23,0x2b]
; CHECK: adds w1, w2, w3, uxtx       ; encoding: [0x41,0x60,0x23,0x2b]
; CHECK: adds w1, w2, w3, sxtb       ; encoding: [0x41,0x80,0x23,0x2b]
; CHECK: adds w1, w2, w3, sxth       ; encoding: [0x41,0xa0,0x23,0x2b]
; CHECK: adds w1, w2, w3, sxtw       ; encoding: [0x41,0xc0,0x23,0x2b]
; CHECK: adds w1, w2, w3, sxtx       ; encoding: [0x41,0xe0,0x23,0x2b]

  adds x1, x2, w3, uxtb
  adds x1, x2, w3, uxth
  adds x1, x2, w3, uxtw
  adds x1, x2, w3, uxtx
  adds x1, x2, w3, sxtb
  adds x1, x2, w3, sxth
  adds x1, x2, w3, sxtw
  adds x1, x2, w3, sxtx

; CHECK: adds x1, x2, w3, uxtb       ; encoding: [0x41,0x00,0x23,0xab]
; CHECK: adds x1, x2, w3, uxth       ; encoding: [0x41,0x20,0x23,0xab]
; CHECK: adds x1, x2, w3, uxtw       ; encoding: [0x41,0x40,0x23,0xab]
; CHECK: adds x1, x2, w3, uxtx       ; encoding: [0x41,0x60,0x23,0xab]
; CHECK: adds x1, x2, w3, sxtb       ; encoding: [0x41,0x80,0x23,0xab]
; CHECK: adds x1, x2, w3, sxth       ; encoding: [0x41,0xa0,0x23,0xab]
; CHECK: adds x1, x2, w3, sxtw       ; encoding: [0x41,0xc0,0x23,0xab]
; CHECK: adds x1, x2, w3, sxtx       ; encoding: [0x41,0xe0,0x23,0xab]

  adds w1, wsp, w3
  adds w1, wsp, w3, uxtw #0
  adds wzr, wsp, w3, lsl #4

; CHECK: adds w1, wsp, w3            ; encoding: [0xe1,0x43,0x23,0x2b]
; CHECK: adds w1, wsp, w3            ; encoding: [0xe1,0x43,0x23,0x2b]
; CHECK: cmn wsp, w3, lsl #4         ; encoding: [0xff,0x53,0x23,0x2b]

  subs w1, w2, w3, uxtb
  subs w1, w2, w3, uxth
  subs w1, w2, w3, uxtw
  subs w1, w2, w3, uxtx
  subs w1, w2, w3, sxtb
  subs w1, w2, w3, sxth
  subs w1, w2, w3, sxtw
  subs w1, w2, w3, sxtx

; CHECK: subs w1, w2, w3, uxtb       ; encoding: [0x41,0x00,0x23,0x6b]
; CHECK: subs w1, w2, w3, uxth       ; encoding: [0x41,0x20,0x23,0x6b]
; CHECK: subs w1, w2, w3, uxtw       ; encoding: [0x41,0x40,0x23,0x6b]
; CHECK: subs w1, w2, w3, uxtx       ; encoding: [0x41,0x60,0x23,0x6b]
; CHECK: subs w1, w2, w3, sxtb       ; encoding: [0x41,0x80,0x23,0x6b]
; CHECK: subs w1, w2, w3, sxth       ; encoding: [0x41,0xa0,0x23,0x6b]
; CHECK: subs w1, w2, w3, sxtw       ; encoding: [0x41,0xc0,0x23,0x6b]
; CHECK: subs w1, w2, w3, sxtx       ; encoding: [0x41,0xe0,0x23,0x6b]

  subs x1, x2, w3, uxtb
  subs x1, x2, w3, uxth
  subs x1, x2, w3, uxtw
  subs x1, x2, w3, uxtx
  subs x1, x2, w3, sxtb
  subs x1, x2, w3, sxth
  subs x1, x2, w3, sxtw
  subs x1, x2, w3, sxtx

; CHECK: subs x1, x2, w3, uxtb       ; encoding: [0x41,0x00,0x23,0xeb]
; CHECK: subs x1, x2, w3, uxth       ; encoding: [0x41,0x20,0x23,0xeb]
; CHECK: subs x1, x2, w3, uxtw       ; encoding: [0x41,0x40,0x23,0xeb]
; CHECK: subs x1, x2, w3, uxtx       ; encoding: [0x41,0x60,0x23,0xeb]
; CHECK: subs x1, x2, w3, sxtb       ; encoding: [0x41,0x80,0x23,0xeb]
; CHECK: subs x1, x2, w3, sxth       ; encoding: [0x41,0xa0,0x23,0xeb]
; CHECK: subs x1, x2, w3, sxtw       ; encoding: [0x41,0xc0,0x23,0xeb]
; CHECK: subs x1, x2, w3, sxtx       ; encoding: [0x41,0xe0,0x23,0xeb]

  subs w1, wsp, w3
  subs w1, wsp, w3, uxtw #0

; CHECK: subs w1, wsp, w3            ; encoding: [0xe1,0x43,0x23,0x6b]
; CHECK: subs w1, wsp, w3            ; encoding: [0xe1,0x43,0x23,0x6b]

  cmp wsp, w9, lsl #0
  subs x3, sp, x9, lsl #2
  cmp wsp, w8, uxtw
  subs wzr, wsp, w8, uxtw
  cmp sp, w8, uxtw
  subs xzr, sp, w8, uxtw

; CHECK: cmp wsp, w9                 ; encoding: [0xff,0x43,0x29,0x6b]
; CHECK: subs x3, sp, x9, lsl #2     ; encoding: [0xe3,0x6b,0x29,0xeb]
; CHECK: cmp wsp, w8                 ; encoding: [0xff,0x43,0x28,0x6b]
; CHECK: cmp wsp, w8                 ; encoding: [0xff,0x43,0x28,0x6b]
; CHECK: cmp sp, w8, uxtw            ; encoding: [0xff,0x43,0x28,0xeb]
; CHECK: cmp sp, w8, uxtw            ; encoding: [0xff,0x43,0x28,0xeb]

  sub wsp, w9, w8, uxtw
  sub w1, wsp, w8, uxtw
  sub wsp, wsp, w8, uxtw
  sub sp, x9, w8, uxtw
  sub x1, sp, w8, uxtw
  sub sp, sp, w8, uxtw
  subs w1, wsp, w8, uxtw
  subs x1, sp, w8, uxtw

; CHECK: sub wsp, w9, w8             ; encoding: [0x3f,0x41,0x28,0x4b]
; CHECK: sub w1, wsp, w8             ; encoding: [0xe1,0x43,0x28,0x4b]
; CHECK: sub wsp, wsp, w8            ; encoding: [0xff,0x43,0x28,0x4b]
; CHECK: sub sp, x9, w8, uxtw        ; encoding: [0x3f,0x41,0x28,0xcb]
; CHECK: sub x1, sp, w8, uxtw        ; encoding: [0xe1,0x43,0x28,0xcb]
; CHECK: sub sp, sp, w8, uxtw        ; encoding: [0xff,0x43,0x28,0xcb]
; CHECK: subs w1, wsp, w8            ; encoding: [0xe1,0x43,0x28,0x6b]
; CHECK: subs x1, sp, w8, uxtw       ; encoding: [0xe1,0x43,0x28,0xeb]

;==---------------------------------------------------------------------------==
; Signed/Unsigned divide
;==---------------------------------------------------------------------------==

  sdiv w1, w2, w3
  sdiv x1, x2, x3
  udiv w1, w2, w3
  udiv x1, x2, x3

; CHECK: sdiv w1, w2, w3             ; encoding: [0x41,0x0c,0xc3,0x1a]
; CHECK: sdiv x1, x2, x3             ; encoding: [0x41,0x0c,0xc3,0x9a]
; CHECK: udiv w1, w2, w3             ; encoding: [0x41,0x08,0xc3,0x1a]
; CHECK: udiv x1, x2, x3             ; encoding: [0x41,0x08,0xc3,0x9a]

;==---------------------------------------------------------------------------==
; Variable shifts
;==---------------------------------------------------------------------------==

  asrv w1, w2, w3
  asrv x1, x2, x3
  asr w1, w2, w3
  asr x1, x2, x3
  lslv w1, w2, w3
  lslv x1, x2, x3
  lsl w1, w2, w3
  lsl x1, x2, x3
  lsrv w1, w2, w3
  lsrv x1, x2, x3
  lsr w1, w2, w3
  lsr x1, x2, x3
  rorv w1, w2, w3
  rorv x1, x2, x3
  ror w1, w2, w3
  ror x1, x2, x3

; CHECK: encoding: [0x41,0x28,0xc3,0x1a]
; CHECK: encoding: [0x41,0x28,0xc3,0x9a]
; CHECK: encoding: [0x41,0x28,0xc3,0x1a]
; CHECK: encoding: [0x41,0x28,0xc3,0x9a]
; CHECK: encoding: [0x41,0x20,0xc3,0x1a]
; CHECK: encoding: [0x41,0x20,0xc3,0x9a]
; CHECK: encoding: [0x41,0x20,0xc3,0x1a]
; CHECK: encoding: [0x41,0x20,0xc3,0x9a]
; CHECK: encoding: [0x41,0x24,0xc3,0x1a]
; CHECK: encoding: [0x41,0x24,0xc3,0x9a]
; CHECK: encoding: [0x41,0x24,0xc3,0x1a]
; CHECK: encoding: [0x41,0x24,0xc3,0x9a]
; CHECK: encoding: [0x41,0x2c,0xc3,0x1a]
; CHECK: encoding: [0x41,0x2c,0xc3,0x9a]
; CHECK: encoding: [0x41,0x2c,0xc3,0x1a]
; CHECK: encoding: [0x41,0x2c,0xc3,0x9a]

;==---------------------------------------------------------------------------==
; One operand instructions
;==---------------------------------------------------------------------------==

  cls w1, w2
  cls x1, x2
  clz w1, w2
  clz x1, x2
  rbit w1, w2
  rbit x1, x2
  rev w1, w2
  rev x1, x2
  rev16 w1, w2
  rev16 x1, x2
  rev32 x1, x2

; CHECK: encoding: [0x41,0x14,0xc0,0x5a]
; CHECK: encoding: [0x41,0x14,0xc0,0xda]
; CHECK: encoding: [0x41,0x10,0xc0,0x5a]
; CHECK: encoding: [0x41,0x10,0xc0,0xda]
; CHECK: encoding: [0x41,0x00,0xc0,0x5a]
; CHECK: encoding: [0x41,0x00,0xc0,0xda]
; CHECK: encoding: [0x41,0x08,0xc0,0x5a]
; CHECK: encoding: [0x41,0x0c,0xc0,0xda]
; CHECK: encoding: [0x41,0x04,0xc0,0x5a]
; CHECK: encoding: [0x41,0x04,0xc0,0xda]
; CHECK: encoding: [0x41,0x08,0xc0,0xda]

;==---------------------------------------------------------------------------==
; 6.6.1 Multiply-add instructions
;==---------------------------------------------------------------------------==

  madd   w1, w2, w3, w4
  madd   x1, x2, x3, x4
  msub   w1, w2, w3, w4
  msub   x1, x2, x3, x4
  smaddl x1, w2, w3, x4
  smsubl x1, w2, w3, x4
  umaddl x1, w2, w3, x4
  umsubl x1, w2, w3, x4

; CHECK: madd   w1, w2, w3, w4       ; encoding: [0x41,0x10,0x03,0x1b]
; CHECK: madd   x1, x2, x3, x4       ; encoding: [0x41,0x10,0x03,0x9b]
; CHECK: msub   w1, w2, w3, w4       ; encoding: [0x41,0x90,0x03,0x1b]
; CHECK: msub   x1, x2, x3, x4       ; encoding: [0x41,0x90,0x03,0x9b]
; CHECK: smaddl x1, w2, w3, x4       ; encoding: [0x41,0x10,0x23,0x9b]
; CHECK: smsubl x1, w2, w3, x4       ; encoding: [0x41,0x90,0x23,0x9b]
; CHECK: umaddl x1, w2, w3, x4       ; encoding: [0x41,0x10,0xa3,0x9b]
; CHECK: umsubl x1, w2, w3, x4       ; encoding: [0x41,0x90,0xa3,0x9b]

;==---------------------------------------------------------------------------==
; Multiply-high instructions
;==---------------------------------------------------------------------------==

  smulh x1, x2, x3
  umulh x1, x2, x3

; CHECK: smulh x1, x2, x3            ; encoding: [0x41,0x7c,0x43,0x9b]
; CHECK: umulh x1, x2, x3            ; encoding: [0x41,0x7c,0xc3,0x9b]

;==---------------------------------------------------------------------------==
; Move immediate instructions
;==---------------------------------------------------------------------------==

  movz w0, #1
  movz x0, #1
  movz w0, #1, lsl #16
  movz x0, #1, lsl #16

; CHECK: mov w0, #1                 ; encoding: [0x20,0x00,0x80,0x52]
; CHECK: mov x0, #1                 ; encoding: [0x20,0x00,0x80,0xd2]
; CHECK: mov w0, #65536             ; encoding: [0x20,0x00,0xa0,0x52]
; CHECK: mov x0, #65536             ; encoding: [0x20,0x00,0xa0,0xd2]

  movn w0, #2
  movn x0, #2
  movn w0, #2, lsl #16
  movn x0, #2, lsl #16

; CHECK: mov w0, #-3                 ; encoding: [0x40,0x00,0x80,0x12]
; CHECK: mov x0, #-3                 ; encoding: [0x40,0x00,0x80,0x92]
; CHECK: mov w0, #-131073            ; encoding: [0x40,0x00,0xa0,0x12]
; CHECK: mov x0, #-131073            ; encoding: [0x40,0x00,0xa0,0x92]

  movk w0, #1
  movk x0, #1
  movk w0, #1, lsl #16
  movk x0, #1, lsl #16

; CHECK: movk w0, #1                 ; encoding: [0x20,0x00,0x80,0x72]
; CHECK: movk x0, #1                 ; encoding: [0x20,0x00,0x80,0xf2]
; CHECK: movk w0, #1, lsl #16        ; encoding: [0x20,0x00,0xa0,0x72]
; CHECK: movk x0, #1, lsl #16        ; encoding: [0x20,0x00,0xa0,0xf2]

;==---------------------------------------------------------------------------==
; Conditionally set flags instructions
;==---------------------------------------------------------------------------==

  ccmn w1, #2, #3, eq
  ccmn x1, #2, #3, eq
  ccmp w1, #2, #3, eq
  ccmp x1, #2, #3, eq

; CHECK: encoding: [0x23,0x08,0x42,0x3a]
; CHECK: encoding: [0x23,0x08,0x42,0xba]
; CHECK: encoding: [0x23,0x08,0x42,0x7a]
; CHECK: encoding: [0x23,0x08,0x42,0xfa]

  ccmn w1, w2, #3, eq
  ccmn x1, x2, #3, eq
  ccmp w1, w2, #3, eq
  ccmp x1, x2, #3, eq

; CHECK: encoding: [0x23,0x00,0x42,0x3a]
; CHECK: encoding: [0x23,0x00,0x42,0xba]
; CHECK: encoding: [0x23,0x00,0x42,0x7a]
; CHECK: encoding: [0x23,0x00,0x42,0xfa]

;==---------------------------------------------------------------------------==
; Conditional select instructions
;==---------------------------------------------------------------------------==

  csel w1, w2, w3, eq
  csel x1, x2, x3, eq
  csinc w1, w2, w3, eq
  csinc x1, x2, x3, eq
  csinv w1, w2, w3, eq
  csinv x1, x2, x3, eq
  csneg w1, w2, w3, eq
  csneg x1, x2, x3, eq

; CHECK: encoding: [0x41,0x00,0x83,0x1a]
; CHECK: encoding: [0x41,0x00,0x83,0x9a]
; CHECK: encoding: [0x41,0x04,0x83,0x1a]
; CHECK: encoding: [0x41,0x04,0x83,0x9a]
; CHECK: encoding: [0x41,0x00,0x83,0x5a]
; CHECK: encoding: [0x41,0x00,0x83,0xda]
; CHECK: encoding: [0x41,0x04,0x83,0x5a]
; CHECK: encoding: [0x41,0x04,0x83,0xda]

; Make sure we handle upper case, too. In particular, condition codes.
  CSEL W16, W7, W27, EQ
  CSEL W15, W6, W26, NE
  CSEL W14, W5, W25, CS
  CSEL W13, W4, W24, HS
  csel w12, w3, w23, CC
  csel w11, w2, w22, LO
  csel w10, w1, w21, MI
  csel x9, x9, x1, PL
  csel x8, x8, x2, VS
  CSEL X7, X7, X3, VC
  CSEL X6, X7, X4, HI
  CSEL X5, X6, X5, LS
  CSEL X4, X5, X6, GE
  csel x3, x4, x7, LT
  csel x2, x3, x8, GT
  csel x1, x2, x9, LE
  csel x10, x1, x20, AL

; CHECK: csel	w16, w7, w27, eq        ; encoding: [0xf0,0x00,0x9b,0x1a]
; CHECK: csel	w15, w6, w26, ne        ; encoding: [0xcf,0x10,0x9a,0x1a]
; CHECK: csel	w14, w5, w25, hs        ; encoding: [0xae,0x20,0x99,0x1a]
; CHECK: csel	w13, w4, w24, hs        ; encoding: [0x8d,0x20,0x98,0x1a]
; CHECK: csel	w12, w3, w23, lo        ; encoding: [0x6c,0x30,0x97,0x1a]
; CHECK: csel	w11, w2, w22, lo        ; encoding: [0x4b,0x30,0x96,0x1a]
; CHECK: csel	w10, w1, w21, mi        ; encoding: [0x2a,0x40,0x95,0x1a]
; CHECK: csel	x9, x9, x1, pl          ; encoding: [0x29,0x51,0x81,0x9a]
; CHECK: csel	x8, x8, x2, vs          ; encoding: [0x08,0x61,0x82,0x9a]
; CHECK: csel	x7, x7, x3, vc          ; encoding: [0xe7,0x70,0x83,0x9a]
; CHECK: csel	x6, x7, x4, hi          ; encoding: [0xe6,0x80,0x84,0x9a]
; CHECK: csel	x5, x6, x5, ls          ; encoding: [0xc5,0x90,0x85,0x9a]
; CHECK: csel	x4, x5, x6, ge          ; encoding: [0xa4,0xa0,0x86,0x9a]
; CHECK: csel	x3, x4, x7, lt          ; encoding: [0x83,0xb0,0x87,0x9a]
; CHECK: csel	x2, x3, x8, gt          ; encoding: [0x62,0xc0,0x88,0x9a]
; CHECK: csel	x1, x2, x9, le          ; encoding: [0x41,0xd0,0x89,0x9a]
; CHECK: csel	x10, x1, x20, al        ; encoding: [0x2a,0xe0,0x94,0x9a]


;==---------------------------------------------------------------------------==
; Scalar saturating arithmetic
;==---------------------------------------------------------------------------==
  uqxtn b4, h2
  uqxtn h2, s3
  uqxtn s9, d2

; CHECK: uqxtn b4, h2                  ; encoding: [0x44,0x48,0x21,0x7e]
; CHECK: uqxtn h2, s3                  ; encoding: [0x62,0x48,0x61,0x7e]
; CHECK: uqxtn s9, d2                  ; encoding: [0x49,0x48,0xa1,0x7e]
