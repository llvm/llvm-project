; RUN: llvm-mc -triple arm64-apple-darwin -mattr=neon -show-encoding -output-asm-variant=1 < %s | FileCheck %s

foo:
;-----------------------------------------------------------------------------
; Floating-point arithmetic
;-----------------------------------------------------------------------------

  fabs s1, s2
  fabs d1, d2

; CHECK: fabs s1, s2                 ; encoding: [0x41,0xc0,0x20,0x1e]
; CHECK: fabs d1, d2                 ; encoding: [0x41,0xc0,0x60,0x1e]

  fadd s1, s2, s3
  fadd d1, d2, d3

; CHECK: fadd s1, s2, s3             ; encoding: [0x41,0x28,0x23,0x1e]
; CHECK: fadd d1, d2, d3             ; encoding: [0x41,0x28,0x63,0x1e]

  fdiv s1, s2, s3
  fdiv d1, d2, d3

; CHECK: fdiv s1, s2, s3             ; encoding: [0x41,0x18,0x23,0x1e]
; CHECK: fdiv d1, d2, d3             ; encoding: [0x41,0x18,0x63,0x1e]

  fmadd s1, s2, s3, s4
  fmadd d1, d2, d3, d4

; CHECK: fmadd s1, s2, s3, s4        ; encoding: [0x41,0x10,0x03,0x1f]
; CHECK: fmadd d1, d2, d3, d4        ; encoding: [0x41,0x10,0x43,0x1f]

  fmax   s1, s2, s3
  fmax   d1, d2, d3
  fmaxnm s1, s2, s3
  fmaxnm d1, d2, d3

; CHECK: fmax   s1, s2, s3           ; encoding: [0x41,0x48,0x23,0x1e]
; CHECK: fmax   d1, d2, d3           ; encoding: [0x41,0x48,0x63,0x1e]
; CHECK: fmaxnm s1, s2, s3           ; encoding: [0x41,0x68,0x23,0x1e]
; CHECK: fmaxnm d1, d2, d3           ; encoding: [0x41,0x68,0x63,0x1e]

  fmin   s1, s2, s3
  fmin   d1, d2, d3
  fminnm s1, s2, s3
  fminnm d1, d2, d3

; CHECK: fmin   s1, s2, s3           ; encoding: [0x41,0x58,0x23,0x1e]
; CHECK: fmin   d1, d2, d3           ; encoding: [0x41,0x58,0x63,0x1e]
; CHECK: fminnm s1, s2, s3           ; encoding: [0x41,0x78,0x23,0x1e]
; CHECK: fminnm d1, d2, d3           ; encoding: [0x41,0x78,0x63,0x1e]

  fmsub s1, s2, s3, s4
  fmsub d1, d2, d3, d4

; CHECK: fmsub s1, s2, s3, s4        ; encoding: [0x41,0x90,0x03,0x1f]
; CHECK: fmsub d1, d2, d3, d4        ; encoding: [0x41,0x90,0x43,0x1f]

  fmul s1, s2, s3
  fmul d1, d2, d3

; CHECK: fmul s1, s2, s3             ; encoding: [0x41,0x08,0x23,0x1e]
; CHECK: fmul d1, d2, d3             ; encoding: [0x41,0x08,0x63,0x1e]

  fneg s1, s2
  fneg d1, d2

; CHECK: fneg s1, s2                 ; encoding: [0x41,0x40,0x21,0x1e]
; CHECK: fneg d1, d2                 ; encoding: [0x41,0x40,0x61,0x1e]

  fnmadd s1, s2, s3, s4
  fnmadd d1, d2, d3, d4

; CHECK: fnmadd s1, s2, s3, s4       ; encoding: [0x41,0x10,0x23,0x1f]
; CHECK: fnmadd d1, d2, d3, d4       ; encoding: [0x41,0x10,0x63,0x1f]

  fnmsub s1, s2, s3, s4
  fnmsub d1, d2, d3, d4

; CHECK: fnmsub s1, s2, s3, s4       ; encoding: [0x41,0x90,0x23,0x1f]
; CHECK: fnmsub d1, d2, d3, d4       ; encoding: [0x41,0x90,0x63,0x1f]

  fnmul s1, s2, s3
  fnmul d1, d2, d3

; CHECK: fnmul s1, s2, s3            ; encoding: [0x41,0x88,0x23,0x1e]
; CHECK: fnmul d1, d2, d3            ; encoding: [0x41,0x88,0x63,0x1e]

  fsqrt s1, s2
  fsqrt d1, d2

; CHECK: fsqrt s1, s2                ; encoding: [0x41,0xc0,0x21,0x1e]
; CHECK: fsqrt d1, d2                ; encoding: [0x41,0xc0,0x61,0x1e]

  fsub s1, s2, s3
  fsub d1, d2, d3

; CHECK: fsub s1, s2, s3             ; encoding: [0x41,0x38,0x23,0x1e]
; CHECK: fsub d1, d2, d3             ; encoding: [0x41,0x38,0x63,0x1e]

;-----------------------------------------------------------------------------
; Floating-point comparison
;-----------------------------------------------------------------------------

  fccmp  s1, s2, #0, eq
  fccmp  d1, d2, #0, eq
  fccmpe s1, s2, #0, eq
  fccmpe d1, d2, #0, eq

; CHECK: fccmp  s1, s2, #0, eq       ; encoding: [0x20,0x04,0x22,0x1e]
; CHECK: fccmp  d1, d2, #0, eq       ; encoding: [0x20,0x04,0x62,0x1e]
; CHECK: fccmpe s1, s2, #0, eq       ; encoding: [0x30,0x04,0x22,0x1e]
; CHECK: fccmpe d1, d2, #0, eq       ; encoding: [0x30,0x04,0x62,0x1e]

  fcmp  s1, s2
  fcmp  d1, d2
  fcmp  s1, #0.0
  fcmp  d1, #0.0
  fcmpe s1, s2
  fcmpe d1, d2
  fcmpe s1, #0.0
  fcmpe d1, #0.0

; CHECK: fcmp  s1, s2                ; encoding: [0x20,0x20,0x22,0x1e]
; CHECK: fcmp  d1, d2                ; encoding: [0x20,0x20,0x62,0x1e]
; CHECK: fcmp  s1, #0.0              ; encoding: [0x28,0x20,0x20,0x1e]
; CHECK: fcmp  d1, #0.0              ; encoding: [0x28,0x20,0x60,0x1e]
; CHECK: fcmpe s1, s2                ; encoding: [0x30,0x20,0x22,0x1e]
; CHECK: fcmpe d1, d2                ; encoding: [0x30,0x20,0x62,0x1e]
; CHECK: fcmpe s1, #0.0              ; encoding: [0x38,0x20,0x20,0x1e]
; CHECK: fcmpe d1, #0.0              ; encoding: [0x38,0x20,0x60,0x1e]

;-----------------------------------------------------------------------------
; Floating-point conditional select
;-----------------------------------------------------------------------------

  fcsel s1, s2, s3, eq
  fcsel d1, d2, d3, eq

; CHECK: fcsel s1, s2, s3, eq        ; encoding: [0x41,0x0c,0x23,0x1e]
; CHECK: fcsel d1, d2, d3, eq        ; encoding: [0x41,0x0c,0x63,0x1e]

;-----------------------------------------------------------------------------
; Floating-point convert
;-----------------------------------------------------------------------------

  fcvt h1, d2
  fcvt s1, d2
  fcvt d1, h2
  fcvt s1, h2
  fcvt d1, s2
  fcvt h1, s2

; CHECK: fcvt h1, d2                 ; encoding: [0x41,0xc0,0x63,0x1e]
; CHECK: fcvt s1, d2                 ; encoding: [0x41,0x40,0x62,0x1e]
; CHECK: fcvt d1, h2                 ; encoding: [0x41,0xc0,0xe2,0x1e]
; CHECK: fcvt s1, h2                 ; encoding: [0x41,0x40,0xe2,0x1e]
; CHECK: fcvt d1, s2                 ; encoding: [0x41,0xc0,0x22,0x1e]
; CHECK: fcvt h1, s2                 ; encoding: [0x41,0xc0,0x23,0x1e]

  fcvtas w1, d2
  fcvtas x1, d2
  fcvtas w1, s2
  fcvtas x1, s2

; CHECK: fcvtas	w1, d2                  ; encoding: [0x41,0x00,0x64,0x1e]
; CHECK: fcvtas	x1, d2                  ; encoding: [0x41,0x00,0x64,0x9e]
; CHECK: fcvtas	w1, s2                  ; encoding: [0x41,0x00,0x24,0x1e]
; CHECK: fcvtas	x1, s2                  ; encoding: [0x41,0x00,0x24,0x9e]

  fcvtau w1, s2
  fcvtau w1, d2
  fcvtau x1, s2
  fcvtau x1, d2

; CHECK: fcvtau	w1, s2                  ; encoding: [0x41,0x00,0x25,0x1e]
; CHECK: fcvtau	w1, d2                  ; encoding: [0x41,0x00,0x65,0x1e]
; CHECK: fcvtau	x1, s2                  ; encoding: [0x41,0x00,0x25,0x9e]
; CHECK: fcvtau	x1, d2                  ; encoding: [0x41,0x00,0x65,0x9e]

  fcvtms w1, s2
  fcvtms w1, d2
  fcvtms x1, s2
  fcvtms x1, d2

; CHECK: fcvtms	w1, s2                  ; encoding: [0x41,0x00,0x30,0x1e]
; CHECK: fcvtms	w1, d2                  ; encoding: [0x41,0x00,0x70,0x1e]
; CHECK: fcvtms	x1, s2                  ; encoding: [0x41,0x00,0x30,0x9e]
; CHECK: fcvtms	x1, d2                  ; encoding: [0x41,0x00,0x70,0x9e]

  fcvtmu w1, s2
  fcvtmu w1, d2
  fcvtmu x1, s2
  fcvtmu x1, d2

; CHECK: fcvtmu	w1, s2                  ; encoding: [0x41,0x00,0x31,0x1e]
; CHECK: fcvtmu	w1, d2                  ; encoding: [0x41,0x00,0x71,0x1e]
; CHECK: fcvtmu	x1, s2                  ; encoding: [0x41,0x00,0x31,0x9e]
; CHECK: fcvtmu	x1, d2                  ; encoding: [0x41,0x00,0x71,0x9e]

  fcvtns w1, s2
  fcvtns w1, d2
  fcvtns x1, s2
  fcvtns x1, d2

; CHECK: fcvtns	w1, s2                  ; encoding: [0x41,0x00,0x20,0x1e]
; CHECK: fcvtns	w1, d2                  ; encoding: [0x41,0x00,0x60,0x1e]
; CHECK: fcvtns	x1, s2                  ; encoding: [0x41,0x00,0x20,0x9e]
; CHECK: fcvtns	x1, d2                  ; encoding: [0x41,0x00,0x60,0x9e]

  fcvtnu w1, s2
  fcvtnu w1, d2
  fcvtnu x1, s2
  fcvtnu x1, d2

; CHECK: fcvtnu	w1, s2                  ; encoding: [0x41,0x00,0x21,0x1e]
; CHECK: fcvtnu	w1, d2                  ; encoding: [0x41,0x00,0x61,0x1e]
; CHECK: fcvtnu	x1, s2                  ; encoding: [0x41,0x00,0x21,0x9e]
; CHECK: fcvtnu	x1, d2                  ; encoding: [0x41,0x00,0x61,0x9e]

  fcvtps w1, s2
  fcvtps w1, d2
  fcvtps x1, s2
  fcvtps x1, d2

; CHECK: fcvtps	w1, s2                  ; encoding: [0x41,0x00,0x28,0x1e]
; CHECK: fcvtps	w1, d2                  ; encoding: [0x41,0x00,0x68,0x1e]
; CHECK: fcvtps	x1, s2                  ; encoding: [0x41,0x00,0x28,0x9e]
; CHECK: fcvtps	x1, d2                  ; encoding: [0x41,0x00,0x68,0x9e]

  fcvtpu w1, s2
  fcvtpu w1, d2
  fcvtpu x1, s2
  fcvtpu x1, d2

; CHECK: fcvtpu	w1, s2                  ; encoding: [0x41,0x00,0x29,0x1e]
; CHECK: fcvtpu	w1, d2                  ; encoding: [0x41,0x00,0x69,0x1e]
; CHECK: fcvtpu	x1, s2                  ; encoding: [0x41,0x00,0x29,0x9e]
; CHECK: fcvtpu	x1, d2                  ; encoding: [0x41,0x00,0x69,0x9e]

  fcvtzs w1, s2
  fcvtzs w1, s2, #1
  fcvtzs w1, d2
  fcvtzs w1, d2, #1
  fcvtzs x1, s2
  fcvtzs x1, s2, #1
  fcvtzs x1, d2
  fcvtzs x1, d2, #1

; CHECK: fcvtzs	w1, s2                  ; encoding: [0x41,0x00,0x38,0x1e]
; CHECK: fcvtzs	w1, s2, #1              ; encoding: [0x41,0xfc,0x18,0x1e]
; CHECK: fcvtzs	w1, d2                  ; encoding: [0x41,0x00,0x78,0x1e]
; CHECK: fcvtzs	w1, d2, #1              ; encoding: [0x41,0xfc,0x58,0x1e]
; CHECK: fcvtzs	x1, s2                  ; encoding: [0x41,0x00,0x38,0x9e]
; CHECK: fcvtzs	x1, s2, #1              ; encoding: [0x41,0xfc,0x18,0x9e]
; CHECK: fcvtzs	x1, d2                  ; encoding: [0x41,0x00,0x78,0x9e]
; CHECK: fcvtzs	x1, d2, #1              ; encoding: [0x41,0xfc,0x58,0x9e]

  fcvtzu w1, s2
  fcvtzu w1, s2, #1
  fcvtzu w1, d2
  fcvtzu w1, d2, #1
  fcvtzu x1, s2
  fcvtzu x1, s2, #1
  fcvtzu x1, d2
  fcvtzu x1, d2, #1

; CHECK: fcvtzu	w1, s2                  ; encoding: [0x41,0x00,0x39,0x1e]
; CHECK: fcvtzu	w1, s2, #1              ; encoding: [0x41,0xfc,0x19,0x1e]
; CHECK: fcvtzu	w1, d2                  ; encoding: [0x41,0x00,0x79,0x1e]
; CHECK: fcvtzu	w1, d2, #1              ; encoding: [0x41,0xfc,0x59,0x1e]
; CHECK: fcvtzu	x1, s2                  ; encoding: [0x41,0x00,0x39,0x9e]
; CHECK: fcvtzu	x1, s2, #1              ; encoding: [0x41,0xfc,0x19,0x9e]
; CHECK: fcvtzu	x1, d2                  ; encoding: [0x41,0x00,0x79,0x9e]
; CHECK: fcvtzu	x1, d2, #1              ; encoding: [0x41,0xfc,0x59,0x9e]

  scvtf s1, w2
  scvtf s1, w2, #1
  scvtf d1, w2
  scvtf d1, w2, #1
  scvtf s1, x2
  scvtf s1, x2, #1
  scvtf d1, x2
  scvtf d1, x2, #1

; CHECK: scvtf	s1, w2                  ; encoding: [0x41,0x00,0x22,0x1e]
; CHECK: scvtf	s1, w2, #1              ; encoding: [0x41,0xfc,0x02,0x1e]
; CHECK: scvtf	d1, w2                  ; encoding: [0x41,0x00,0x62,0x1e]
; CHECK: scvtf	d1, w2, #1              ; encoding: [0x41,0xfc,0x42,0x1e]
; CHECK: scvtf	s1, x2                  ; encoding: [0x41,0x00,0x22,0x9e]
; CHECK: scvtf	s1, x2, #1              ; encoding: [0x41,0xfc,0x02,0x9e]
; CHECK: scvtf	d1, x2                  ; encoding: [0x41,0x00,0x62,0x9e]
; CHECK: scvtf	d1, x2, #1              ; encoding: [0x41,0xfc,0x42,0x9e]

  ucvtf s1, w2
  ucvtf s1, w2, #1
  ucvtf d1, w2
  ucvtf d1, w2, #1
  ucvtf s1, x2
  ucvtf s1, x2, #1
  ucvtf d1, x2
  ucvtf d1, x2, #1

; CHECK: ucvtf	s1, w2                  ; encoding: [0x41,0x00,0x23,0x1e]
; CHECK: ucvtf	s1, w2, #1              ; encoding: [0x41,0xfc,0x03,0x1e]
; CHECK: ucvtf	d1, w2                  ; encoding: [0x41,0x00,0x63,0x1e]
; CHECK: ucvtf	d1, w2, #1              ; encoding: [0x41,0xfc,0x43,0x1e]
; CHECK: ucvtf	s1, x2                  ; encoding: [0x41,0x00,0x23,0x9e]
; CHECK: ucvtf	s1, x2, #1              ; encoding: [0x41,0xfc,0x03,0x9e]
; CHECK: ucvtf	d1, x2                  ; encoding: [0x41,0x00,0x63,0x9e]
; CHECK: ucvtf	d1, x2, #1              ; encoding: [0x41,0xfc,0x43,0x9e]

;-----------------------------------------------------------------------------
; Floating-point move
;-----------------------------------------------------------------------------

  fmov s1, w2
  fmov w1, s2
  fmov d1, x2
  fmov x1, d2

; CHECK: fmov s1, w2                 ; encoding: [0x41,0x00,0x27,0x1e]
; CHECK: fmov w1, s2                 ; encoding: [0x41,0x00,0x26,0x1e]
; CHECK: fmov d1, x2                 ; encoding: [0x41,0x00,0x67,0x9e]
; CHECK: fmov x1, d2                 ; encoding: [0x41,0x00,0x66,0x9e]

  fmov s1, #0.125
  fmov s1, #0x40
  fmov d1, #0.125
  fmov d1, #0x40
  fmov d1, #-4.843750e-01
  fmov d1, #4.843750e-01
  fmov d3, #3
  fmov s2, #0.0
  fmov d2, #0.0

; CHECK: fmov s1, #0.12500000      ; encoding: [0x01,0x10,0x28,0x1e]
; CHECK: fmov s1, #0.12500000      ; encoding: [0x01,0x10,0x28,0x1e]
; CHECK: fmov d1, #0.12500000      ; encoding: [0x01,0x10,0x68,0x1e]
; CHECK: fmov d1, #0.12500000      ; encoding: [0x01,0x10,0x68,0x1e]
; CHECK: fmov d1, #-0.48437500     ; encoding: [0x01,0xf0,0x7b,0x1e]
; CHECK: fmov d1, #0.48437500      ; encoding: [0x01,0xf0,0x6b,0x1e]
; CHECK: fmov d3, #3.00000000      ; encoding: [0x03,0x10,0x61,0x1e]
; CHECK: fmov s2, wzr                ; encoding: [0xe2,0x03,0x27,0x1e]
; CHECK: fmov d2, xzr                ; encoding: [0xe2,0x03,0x67,0x9e]

  fmov s1, s2
  fmov d1, d2

; CHECK: fmov s1, s2                 ; encoding: [0x41,0x40,0x20,0x1e]
; CHECK: fmov d1, d2                 ; encoding: [0x41,0x40,0x60,0x1e]


  fmov x2, v5.d[1]
  fmov.d x9, v7[1]
  fmov v1.d[1], x1
  fmov.d v8[1], x6

; CHECK: fmov.d	x2, v5[1]               ; encoding: [0xa2,0x00,0xae,0x9e]
; CHECK: fmov.d	x9, v7[1]               ; encoding: [0xe9,0x00,0xae,0x9e]
; CHECK: fmov.d	v1[1], x1               ; encoding: [0x21,0x00,0xaf,0x9e]
; CHECK: fmov.d	v8[1], x6               ; encoding: [0xc8,0x00,0xaf,0x9e]


;-----------------------------------------------------------------------------
; Floating-point round to integral
;-----------------------------------------------------------------------------

  frinta s1, s2
  frinta d1, d2

; CHECK: frinta s1, s2               ; encoding: [0x41,0x40,0x26,0x1e]
; CHECK: frinta d1, d2               ; encoding: [0x41,0x40,0x66,0x1e]

  frinti s1, s2
  frinti d1, d2

; CHECK: frinti s1, s2               ; encoding: [0x41,0xc0,0x27,0x1e]
; CHECK: frinti d1, d2               ; encoding: [0x41,0xc0,0x67,0x1e]

  frintm s1, s2
  frintm d1, d2

; CHECK: frintm s1, s2               ; encoding: [0x41,0x40,0x25,0x1e]
; CHECK: frintm d1, d2               ; encoding: [0x41,0x40,0x65,0x1e]

  frintn s1, s2
  frintn d1, d2

; CHECK: frintn s1, s2               ; encoding: [0x41,0x40,0x24,0x1e]
; CHECK: frintn d1, d2               ; encoding: [0x41,0x40,0x64,0x1e]

  frintp s1, s2
  frintp d1, d2

; CHECK: frintp s1, s2               ; encoding: [0x41,0xc0,0x24,0x1e]
; CHECK: frintp d1, d2               ; encoding: [0x41,0xc0,0x64,0x1e]

  frintx s1, s2
  frintx d1, d2

; CHECK: frintx s1, s2               ; encoding: [0x41,0x40,0x27,0x1e]
; CHECK: frintx d1, d2               ; encoding: [0x41,0x40,0x67,0x1e]

  frintz s1, s2
  frintz d1, d2

; CHECK: frintz s1, s2               ; encoding: [0x41,0xc0,0x25,0x1e]
; CHECK: frintz d1, d2               ; encoding: [0x41,0xc0,0x65,0x1e]

  cmhs d0, d0, d0
  cmtst d0, d0, d0

; CHECK: cmhs	d0, d0, d0              ; encoding: [0x00,0x3c,0xe0,0x7e]
; CHECK: cmtst	d0, d0, d0              ; encoding: [0x00,0x8c,0xe0,0x5e]



;-----------------------------------------------------------------------------
; Floating-point extract and narrow
;-----------------------------------------------------------------------------
  sqxtn b4, h2
  sqxtn h2, s3
  sqxtn s9, d2

; CHECK: sqxtn b4, h2                  ; encoding: [0x44,0x48,0x21,0x5e]
; CHECK: sqxtn h2, s3                  ; encoding: [0x62,0x48,0x61,0x5e]
; CHECK: sqxtn s9, d2                  ; encoding: [0x49,0x48,0xa1,0x5e]

  sqxtun b4, h2
  sqxtun h2, s3
  sqxtun s9, d2

; CHECK: sqxtun b4, h2                  ; encoding: [0x44,0x28,0x21,0x7e]
; CHECK: sqxtun h2, s3                  ; encoding: [0x62,0x28,0x61,0x7e]
; CHECK: sqxtun s9, d2                  ; encoding: [0x49,0x28,0xa1,0x7e]

  uqxtn b4, h2
  uqxtn h2, s3
  uqxtn s9, d2

; CHECK: uqxtn b4, h2                  ; encoding: [0x44,0x48,0x21,0x7e]
; CHECK: uqxtn h2, s3                  ; encoding: [0x62,0x48,0x61,0x7e]
; CHECK: uqxtn s9, d2                  ; encoding: [0x49,0x48,0xa1,0x7e]
