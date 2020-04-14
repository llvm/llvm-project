; RUN: llvm-mc -triple arm64-apple-darwin -show-encoding < %s | FileCheck %s

foo:
;==---------------------------------------------------------------------------==
; 5.4.2 Logical (immediate)
;==---------------------------------------------------------------------------==

  and   w0, w0, #1
  and   x0, x0, #1
  and   w1, w2, #15
  and   x1, x2, #15
  and   sp, x5, #~15
  ands  w0, w0, #1
  ands  x0, x0, #1
  ands  w1, w2, #15
  ands  x1, x2, #15

; CHECK: and  w0, w0, #0x1           ; encoding: [0x00,0x00,0x00,0x12]
; CHECK: and  x0, x0, #0x1           ; encoding: [0x00,0x00,0x40,0x92]
; CHECK: and  w1, w2, #0xf           ; encoding: [0x41,0x0c,0x00,0x12]
; CHECK: and  x1, x2, #0xf           ; encoding: [0x41,0x0c,0x40,0x92]
; CHECK: and  sp, x5, #0xfffffffffffffff0 ; encoding: [0xbf,0xec,0x7c,0x92]
; CHECK: ands w0, w0, #0x1           ; encoding: [0x00,0x00,0x00,0x72]
; CHECK: ands x0, x0, #0x1           ; encoding: [0x00,0x00,0x40,0xf2]
; CHECK: ands w1, w2, #0xf           ; encoding: [0x41,0x0c,0x00,0x72]
; CHECK: ands x1, x2, #0xf           ; encoding: [0x41,0x0c,0x40,0xf2]

  eor w1, w2, #0x4000
  eor x1, x2, #0x8000

; CHECK: eor w1, w2, #0x4000         ; encoding: [0x41,0x00,0x12,0x52]
; CHECK: eor x1, x2, #0x8000         ; encoding: [0x41,0x00,0x71,0xd2]

  orr w1, w2, #0x4000
  orr x1, x2, #0x8000

; CHECK: orr w1, w2, #0x4000         ; encoding: [0x41,0x00,0x12,0x32]
; CHECK: orr x1, x2, #0x8000         ; encoding: [0x41,0x00,0x71,0xb2]

  orr w8, wzr, #0x1
  orr x8, xzr, #0x1

; CHECK: orr w8, wzr, #0x1           ; encoding: [0xe8,0x03,0x00,0x32]
; CHECK: orr x8, xzr, #0x1           ; encoding: [0xe8,0x03,0x40,0xb2]

;==---------------------------------------------------------------------------==
; 5.5.3 Logical (shifted register)
;==---------------------------------------------------------------------------==

  and   w1, w2, w3
  and   x1, x2, x3
  and   w1, w2, w3, lsl #2
  and   x1, x2, x3, lsl #2
  and   w1, w2, w3, lsr #2
  and   x1, x2, x3, lsr #2
  and   w1, w2, w3, asr #2
  and   x1, x2, x3, asr #2
  and   w1, w2, w3, ror #2
  and   x1, x2, x3, ror #2

; CHECK: and  w1, w2, w3             ; encoding: [0x41,0x00,0x03,0x0a]
; CHECK: and  x1, x2, x3             ; encoding: [0x41,0x00,0x03,0x8a]
; CHECK: and  w1, w2, w3, lsl #2     ; encoding: [0x41,0x08,0x03,0x0a]
; CHECK: and  x1, x2, x3, lsl #2     ; encoding: [0x41,0x08,0x03,0x8a]
; CHECK: and  w1, w2, w3, lsr #2     ; encoding: [0x41,0x08,0x43,0x0a]
; CHECK: and  x1, x2, x3, lsr #2     ; encoding: [0x41,0x08,0x43,0x8a]
; CHECK: and  w1, w2, w3, asr #2     ; encoding: [0x41,0x08,0x83,0x0a]
; CHECK: and  x1, x2, x3, asr #2     ; encoding: [0x41,0x08,0x83,0x8a]
; CHECK: and  w1, w2, w3, ror #2     ; encoding: [0x41,0x08,0xc3,0x0a]
; CHECK: and  x1, x2, x3, ror #2     ; encoding: [0x41,0x08,0xc3,0x8a]

  ands  w1, w2, w3
  ands  x1, x2, x3
  ands  w1, w2, w3, lsl #2
  ands  x1, x2, x3, lsl #2
  ands  w1, w2, w3, lsr #2
  ands  x1, x2, x3, lsr #2
  ands  w1, w2, w3, asr #2
  ands  x1, x2, x3, asr #2
  ands  w1, w2, w3, ror #2
  ands  x1, x2, x3, ror #2

; CHECK: ands w1, w2, w3             ; encoding: [0x41,0x00,0x03,0x6a]
; CHECK: ands x1, x2, x3             ; encoding: [0x41,0x00,0x03,0xea]
; CHECK: ands w1, w2, w3, lsl #2     ; encoding: [0x41,0x08,0x03,0x6a]
; CHECK: ands x1, x2, x3, lsl #2     ; encoding: [0x41,0x08,0x03,0xea]
; CHECK: ands w1, w2, w3, lsr #2     ; encoding: [0x41,0x08,0x43,0x6a]
; CHECK: ands x1, x2, x3, lsr #2     ; encoding: [0x41,0x08,0x43,0xea]
; CHECK: ands w1, w2, w3, asr #2     ; encoding: [0x41,0x08,0x83,0x6a]
; CHECK: ands x1, x2, x3, asr #2     ; encoding: [0x41,0x08,0x83,0xea]
; CHECK: ands w1, w2, w3, ror #2     ; encoding: [0x41,0x08,0xc3,0x6a]
; CHECK: ands x1, x2, x3, ror #2     ; encoding: [0x41,0x08,0xc3,0xea]

  bic w1, w2, w3
  bic x1, x2, x3
  bic w1, w2, w3, lsl #3
  bic x1, x2, x3, lsl #3
  bic w1, w2, w3, lsr #3
  bic x1, x2, x3, lsr #3
  bic w1, w2, w3, asr #3
  bic x1, x2, x3, asr #3
  bic w1, w2, w3, ror #3
  bic x1, x2, x3, ror #3

; CHECK: bic w1, w2, w3              ; encoding: [0x41,0x00,0x23,0x0a]
; CHECK: bic x1, x2, x3              ; encoding: [0x41,0x00,0x23,0x8a]
; CHECK: bic w1, w2, w3, lsl #3      ; encoding: [0x41,0x0c,0x23,0x0a]
; CHECK: bic x1, x2, x3, lsl #3      ; encoding: [0x41,0x0c,0x23,0x8a]
; CHECK: bic w1, w2, w3, lsr #3      ; encoding: [0x41,0x0c,0x63,0x0a]
; CHECK: bic x1, x2, x3, lsr #3      ; encoding: [0x41,0x0c,0x63,0x8a]
; CHECK: bic w1, w2, w3, asr #3      ; encoding: [0x41,0x0c,0xa3,0x0a]
; CHECK: bic x1, x2, x3, asr #3      ; encoding: [0x41,0x0c,0xa3,0x8a]
; CHECK: bic w1, w2, w3, ror #3      ; encoding: [0x41,0x0c,0xe3,0x0a]
; CHECK: bic x1, x2, x3, ror #3      ; encoding: [0x41,0x0c,0xe3,0x8a]

  bics w1, w2, w3
  bics x1, x2, x3
  bics w1, w2, w3, lsl #3
  bics x1, x2, x3, lsl #3
  bics w1, w2, w3, lsr #3
  bics x1, x2, x3, lsr #3
  bics w1, w2, w3, asr #3
  bics x1, x2, x3, asr #3
  bics w1, w2, w3, ror #3
  bics x1, x2, x3, ror #3

; CHECK: bics w1, w2, w3             ; encoding: [0x41,0x00,0x23,0x6a]
; CHECK: bics x1, x2, x3             ; encoding: [0x41,0x00,0x23,0xea]
; CHECK: bics w1, w2, w3, lsl #3     ; encoding: [0x41,0x0c,0x23,0x6a]
; CHECK: bics x1, x2, x3, lsl #3     ; encoding: [0x41,0x0c,0x23,0xea]
; CHECK: bics w1, w2, w3, lsr #3     ; encoding: [0x41,0x0c,0x63,0x6a]
; CHECK: bics x1, x2, x3, lsr #3     ; encoding: [0x41,0x0c,0x63,0xea]
; CHECK: bics w1, w2, w3, asr #3     ; encoding: [0x41,0x0c,0xa3,0x6a]
; CHECK: bics x1, x2, x3, asr #3     ; encoding: [0x41,0x0c,0xa3,0xea]
; CHECK: bics w1, w2, w3, ror #3     ; encoding: [0x41,0x0c,0xe3,0x6a]
; CHECK: bics x1, x2, x3, ror #3     ; encoding: [0x41,0x0c,0xe3,0xea]

  eon w1, w2, w3
  eon x1, x2, x3
  eon w1, w2, w3, lsl #4
  eon x1, x2, x3, lsl #4
  eon w1, w2, w3, lsr #4
  eon x1, x2, x3, lsr #4
  eon w1, w2, w3, asr #4
  eon x1, x2, x3, asr #4
  eon w1, w2, w3, ror #4
  eon x1, x2, x3, ror #4

; CHECK: eon w1, w2, w3              ; encoding: [0x41,0x00,0x23,0x4a]
; CHECK: eon x1, x2, x3              ; encoding: [0x41,0x00,0x23,0xca]
; CHECK: eon w1, w2, w3, lsl #4      ; encoding: [0x41,0x10,0x23,0x4a]
; CHECK: eon x1, x2, x3, lsl #4      ; encoding: [0x41,0x10,0x23,0xca]
; CHECK: eon w1, w2, w3, lsr #4      ; encoding: [0x41,0x10,0x63,0x4a]
; CHECK: eon x1, x2, x3, lsr #4      ; encoding: [0x41,0x10,0x63,0xca]
; CHECK: eon w1, w2, w3, asr #4      ; encoding: [0x41,0x10,0xa3,0x4a]
; CHECK: eon x1, x2, x3, asr #4      ; encoding: [0x41,0x10,0xa3,0xca]
; CHECK: eon w1, w2, w3, ror #4      ; encoding: [0x41,0x10,0xe3,0x4a]
; CHECK: eon x1, x2, x3, ror #4      ; encoding: [0x41,0x10,0xe3,0xca]

  eor w1, w2, w3
  eor x1, x2, x3
  eor w1, w2, w3, lsl #5
  eor x1, x2, x3, lsl #5
  eor w1, w2, w3, lsr #5
  eor x1, x2, x3, lsr #5
  eor w1, w2, w3, asr #5
  eor x1, x2, x3, asr #5
  eor w1, w2, w3, ror #5
  eor x1, x2, x3, ror #5

; CHECK: eor w1, w2, w3              ; encoding: [0x41,0x00,0x03,0x4a]
; CHECK: eor x1, x2, x3              ; encoding: [0x41,0x00,0x03,0xca]
; CHECK: eor w1, w2, w3, lsl #5      ; encoding: [0x41,0x14,0x03,0x4a]
; CHECK: eor x1, x2, x3, lsl #5      ; encoding: [0x41,0x14,0x03,0xca]
; CHECK: eor w1, w2, w3, lsr #5      ; encoding: [0x41,0x14,0x43,0x4a]
; CHECK: eor x1, x2, x3, lsr #5      ; encoding: [0x41,0x14,0x43,0xca]
; CHECK: eor w1, w2, w3, asr #5      ; encoding: [0x41,0x14,0x83,0x4a]
; CHECK: eor x1, x2, x3, asr #5      ; encoding: [0x41,0x14,0x83,0xca]
; CHECK: eor w1, w2, w3, ror #5      ; encoding: [0x41,0x14,0xc3,0x4a]
; CHECK: eor x1, x2, x3, ror #5      ; encoding: [0x41,0x14,0xc3,0xca]

  orr w1, w2, w3
  orr x1, x2, x3
  orr w1, w2, w3, lsl #6
  orr x1, x2, x3, lsl #6
  orr w1, w2, w3, lsr #6
  orr x1, x2, x3, lsr #6
  orr w1, w2, w3, asr #6
  orr x1, x2, x3, asr #6
  orr w1, w2, w3, ror #6
  orr x1, x2, x3, ror #6

; CHECK: orr w1, w2, w3              ; encoding: [0x41,0x00,0x03,0x2a]
; CHECK: orr x1, x2, x3              ; encoding: [0x41,0x00,0x03,0xaa]
; CHECK: orr w1, w2, w3, lsl #6      ; encoding: [0x41,0x18,0x03,0x2a]
; CHECK: orr x1, x2, x3, lsl #6      ; encoding: [0x41,0x18,0x03,0xaa]
; CHECK: orr w1, w2, w3, lsr #6      ; encoding: [0x41,0x18,0x43,0x2a]
; CHECK: orr x1, x2, x3, lsr #6      ; encoding: [0x41,0x18,0x43,0xaa]
; CHECK: orr w1, w2, w3, asr #6      ; encoding: [0x41,0x18,0x83,0x2a]
; CHECK: orr x1, x2, x3, asr #6      ; encoding: [0x41,0x18,0x83,0xaa]
; CHECK: orr w1, w2, w3, ror #6      ; encoding: [0x41,0x18,0xc3,0x2a]
; CHECK: orr x1, x2, x3, ror #6      ; encoding: [0x41,0x18,0xc3,0xaa]

  orn w1, w2, w3
  orn x1, x2, x3
  orn w1, w2, w3, lsl #7
  orn x1, x2, x3, lsl #7
  orn w1, w2, w3, lsr #7
  orn x1, x2, x3, lsr #7
  orn w1, w2, w3, asr #7
  orn x1, x2, x3, asr #7
  orn w1, w2, w3, ror #7
  orn x1, x2, x3, ror #7

; CHECK: orn w1, w2, w3              ; encoding: [0x41,0x00,0x23,0x2a]
; CHECK: orn x1, x2, x3              ; encoding: [0x41,0x00,0x23,0xaa]
; CHECK: orn w1, w2, w3, lsl #7      ; encoding: [0x41,0x1c,0x23,0x2a]
; CHECK: orn x1, x2, x3, lsl #7      ; encoding: [0x41,0x1c,0x23,0xaa]
; CHECK: orn w1, w2, w3, lsr #7      ; encoding: [0x41,0x1c,0x63,0x2a]
; CHECK: orn x1, x2, x3, lsr #7      ; encoding: [0x41,0x1c,0x63,0xaa]
; CHECK: orn w1, w2, w3, asr #7      ; encoding: [0x41,0x1c,0xa3,0x2a]
; CHECK: orn x1, x2, x3, asr #7      ; encoding: [0x41,0x1c,0xa3,0xaa]
; CHECK: orn w1, w2, w3, ror #7      ; encoding: [0x41,0x1c,0xe3,0x2a]
; CHECK: orn x1, x2, x3, ror #7      ; encoding: [0x41,0x1c,0xe3,0xaa]

;; Allow all-1 in top bits.
  and w0, w0, #~(0xfe<<24)
  and w1, w1, #~(0xff<<24)

; CHECK: and w0, w0, #0x1ffffff
; CHECK: and w1, w1, #0xffffff
