; RUN: not llvm-mc -triple arm64-apple-darwin -mattr=neon -show-encoding -output-asm-variant=1 < %s 2>%t | FileCheck %s
; RUN: FileCheck %s < %t --check-prefix=NO-FP16
; RUN: llvm-mc -triple arm64-apple-darwin -mattr=neon,v8.2a,fullfp16 -show-encoding -output-asm-variant=1 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=FP16

foo:
;-----------------------------------------------------------------------------
; Floating-point arithmetic
;-----------------------------------------------------------------------------

  fabs h1, h2
  fabs s1, s2
  fabs d1, d2

; FP16:  fabs h1, h2                 ; encoding: [0x41,0xc0,0xe0,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT:  fabs h1, h2
; CHECK: fabs s1, s2                 ; encoding: [0x41,0xc0,0x20,0x1e]
; CHECK: fabs d1, d2                 ; encoding: [0x41,0xc0,0x60,0x1e]

  fadd h1, h2, h3
  fadd s1, s2, s3
  fadd d1, d2, d3

; FP16:  fadd h1, h2, h3             ; encoding: [0x41,0x28,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT:  fadd h1, h2, h3
; CHECK: fadd s1, s2, s3             ; encoding: [0x41,0x28,0x23,0x1e]
; CHECK: fadd d1, d2, d3             ; encoding: [0x41,0x28,0x63,0x1e]

  fdiv h1, h2, h3
  fdiv s1, s2, s3
  fdiv d1, d2, d3

; FP16:  fdiv h1, h2, h3             ; encoding: [0x41,0x18,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT:  fdiv h1, h2, h3
; CHECK: fdiv s1, s2, s3             ; encoding: [0x41,0x18,0x23,0x1e]
; CHECK: fdiv d1, d2, d3             ; encoding: [0x41,0x18,0x63,0x1e]

  fmadd h1, h2, h3, h4
  fmadd s1, s2, s3, s4
  fmadd d1, d2, d3, d4

; FP16:  fmadd h1, h2, h3, h4        ; encoding: [0x41,0x10,0xc3,0x1f]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT:  fmadd h1, h2, h3, h4
; CHECK: fmadd s1, s2, s3, s4        ; encoding: [0x41,0x10,0x03,0x1f]
; CHECK: fmadd d1, d2, d3, d4        ; encoding: [0x41,0x10,0x43,0x1f]

  fmax   h1, h2, h3
  fmax   s1, s2, s3
  fmax   d1, d2, d3
  fmaxnm h1, h2, h3
  fmaxnm s1, s2, s3
  fmaxnm d1, d2, d3

; FP16:  fmax   h1, h2, h3           ; encoding: [0x41,0x48,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmax   h1, h2, h3
; CHECK: fmax   s1, s2, s3           ; encoding: [0x41,0x48,0x23,0x1e]
; CHECK: fmax   d1, d2, d3           ; encoding: [0x41,0x48,0x63,0x1e]
; FP16:  fmaxnm h1, h2, h3           ; encoding: [0x41,0x68,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmaxnm h1, h2, h3
; CHECK: fmaxnm s1, s2, s3           ; encoding: [0x41,0x68,0x23,0x1e]
; CHECK: fmaxnm d1, d2, d3           ; encoding: [0x41,0x68,0x63,0x1e]

  fmin   h1, h2, h3
  fmin   s1, s2, s3
  fmin   d1, d2, d3
  fminnm h1, h2, h3
  fminnm s1, s2, s3
  fminnm d1, d2, d3

; FP16:  fmin   h1, h2, h3           ; encoding: [0x41,0x58,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmin   h1, h2, h3
; CHECK: fmin   s1, s2, s3           ; encoding: [0x41,0x58,0x23,0x1e]
; CHECK: fmin   d1, d2, d3           ; encoding: [0x41,0x58,0x63,0x1e]
; FP16:  fminnm h1, h2, h3           ; encoding: [0x41,0x78,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fminnm h1, h2, h3
; CHECK: fminnm s1, s2, s3           ; encoding: [0x41,0x78,0x23,0x1e]
; CHECK: fminnm d1, d2, d3           ; encoding: [0x41,0x78,0x63,0x1e]

  fmsub h1, h2, h3, h4
  fmsub s1, s2, s3, s4
  fmsub d1, d2, d3, d4

; FP16:  fmsub h1, h2, h3, h4        ; encoding: [0x41,0x90,0xc3,0x1f]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmsub h1, h2, h3, h4
; CHECK: fmsub s1, s2, s3, s4        ; encoding: [0x41,0x90,0x03,0x1f]
; CHECK: fmsub d1, d2, d3, d4        ; encoding: [0x41,0x90,0x43,0x1f]

  fmul h1, h2, h3
  fmul s1, s2, s3
  fmul d1, d2, d3

; FP16:  fmul h1, h2, h3             ; encoding: [0x41,0x08,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmul h1, h2, h3
; CHECK: fmul s1, s2, s3             ; encoding: [0x41,0x08,0x23,0x1e]
; CHECK: fmul d1, d2, d3             ; encoding: [0x41,0x08,0x63,0x1e]

  fneg h1, h2
  fneg s1, s2
  fneg d1, d2

; FP16:  fneg h1, h2                 ; encoding: [0x41,0x40,0xe1,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fneg h1, h2
; CHECK: fneg s1, s2                 ; encoding: [0x41,0x40,0x21,0x1e]
; CHECK: fneg d1, d2                 ; encoding: [0x41,0x40,0x61,0x1e]

  fnmadd h1, h2, h3, h4
  fnmadd s1, s2, s3, s4
  fnmadd d1, d2, d3, d4

; FP16:  fnmadd h1, h2, h3, h4       ; encoding: [0x41,0x10,0xe3,0x1f]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fnmadd h1, h2, h3, h4
; CHECK: fnmadd s1, s2, s3, s4       ; encoding: [0x41,0x10,0x23,0x1f]
; CHECK: fnmadd d1, d2, d3, d4       ; encoding: [0x41,0x10,0x63,0x1f]

  fnmsub h1, h2, h3, h4
  fnmsub s1, s2, s3, s4
  fnmsub d1, d2, d3, d4

; FP16:  fnmsub h1, h2, h3, h4       ; encoding: [0x41,0x90,0xe3,0x1f]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fnmsub h1, h2, h3, h4
; CHECK: fnmsub s1, s2, s3, s4       ; encoding: [0x41,0x90,0x23,0x1f]
; CHECK: fnmsub d1, d2, d3, d4       ; encoding: [0x41,0x90,0x63,0x1f]

  fnmul h1, h2, h3
  fnmul s1, s2, s3
  fnmul d1, d2, d3

; FP16:  fnmul h1, h2, h3            ; encoding: [0x41,0x88,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fnmul h1, h2, h3
; CHECK: fnmul s1, s2, s3            ; encoding: [0x41,0x88,0x23,0x1e]
; CHECK: fnmul d1, d2, d3            ; encoding: [0x41,0x88,0x63,0x1e]

  fsqrt h1, h2
  fsqrt s1, s2
  fsqrt d1, d2

; FP16:  fsqrt h1, h2                ; encoding: [0x41,0xc0,0xe1,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fsqrt h1, h2
; CHECK: fsqrt s1, s2                ; encoding: [0x41,0xc0,0x21,0x1e]
; CHECK: fsqrt d1, d2                ; encoding: [0x41,0xc0,0x61,0x1e]

  fsub h1, h2, h3
  fsub s1, s2, s3
  fsub d1, d2, d3

; FP16:  fsub h1, h2, h3             ; encoding: [0x41,0x38,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fsub h1, h2, h3
; CHECK: fsub s1, s2, s3             ; encoding: [0x41,0x38,0x23,0x1e]
; CHECK: fsub d1, d2, d3             ; encoding: [0x41,0x38,0x63,0x1e]

;-----------------------------------------------------------------------------
; Floating-point comparison
;-----------------------------------------------------------------------------

  fccmp  h1, h2, #0, eq
  fccmp  s1, s2, #0, eq
  fccmp  d1, d2, #0, eq
  fccmpe h1, h2, #0, eq
  fccmpe s1, s2, #0, eq
  fccmpe d1, d2, #0, eq

; FP16:  fccmp  h1, h2, #0, eq       ; encoding: [0x20,0x04,0xe2,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fccmp  h1, h2, #0, eq
; CHECK: fccmp  s1, s2, #0, eq       ; encoding: [0x20,0x04,0x22,0x1e]
; CHECK: fccmp  d1, d2, #0, eq       ; encoding: [0x20,0x04,0x62,0x1e]
; FP16:  fccmpe h1, h2, #0, eq       ; encoding: [0x30,0x04,0xe2,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fccmpe h1, h2, #0, eq
; CHECK: fccmpe s1, s2, #0, eq       ; encoding: [0x30,0x04,0x22,0x1e]
; CHECK: fccmpe d1, d2, #0, eq       ; encoding: [0x30,0x04,0x62,0x1e]

  fcmp  h1, h2
  fcmp  s1, s2
  fcmp  d1, d2
  fcmp  h1, #0.0
  fcmp  s1, #0.0
  fcmp  d1, #0.0
  fcmpe h1, h2
  fcmpe s1, s2
  fcmpe d1, d2
  fcmpe h1, #0.0
  fcmpe s1, #0.0
  fcmpe d1, #0.0

; FP16:  fcmp  h1, h2                ; encoding: [0x20,0x20,0xe2,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcmp  h1, h2
; CHECK: fcmp  s1, s2                ; encoding: [0x20,0x20,0x22,0x1e]
; CHECK: fcmp  d1, d2                ; encoding: [0x20,0x20,0x62,0x1e]
; FP16:  fcmp  h1, #0.0              ; encoding: [0x28,0x20,0xe0,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcmp  h1, #0.0
; CHECK: fcmp  s1, #0.0              ; encoding: [0x28,0x20,0x20,0x1e]
; CHECK: fcmp  d1, #0.0              ; encoding: [0x28,0x20,0x60,0x1e]
; FP16:  fcmpe h1, h2                ; encoding: [0x30,0x20,0xe2,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcmpe h1, h2
; CHECK: fcmpe s1, s2                ; encoding: [0x30,0x20,0x22,0x1e]
; CHECK: fcmpe d1, d2                ; encoding: [0x30,0x20,0x62,0x1e]
; FP16:  fcmpe h1, #0.0              ; encoding: [0x38,0x20,0xe0,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcmpe h1, #0.0
; CHECK: fcmpe s1, #0.0              ; encoding: [0x38,0x20,0x20,0x1e]
; CHECK: fcmpe d1, #0.0              ; encoding: [0x38,0x20,0x60,0x1e]

;-----------------------------------------------------------------------------
; Floating-point conditional select
;-----------------------------------------------------------------------------

  fcsel h1, h2, h3, eq
  fcsel s1, s2, s3, eq
  fcsel d1, d2, d3, eq

; FP16:  fcsel h1, h2, h3, eq        ; encoding: [0x41,0x0c,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcsel h1, h2, h3, eq
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
  fcvtas w1, h2
  fcvtas x1, h2

; CHECK: fcvtas w1, d2                  ; encoding: [0x41,0x00,0x64,0x1e]
; CHECK: fcvtas x1, d2                  ; encoding: [0x41,0x00,0x64,0x9e]
; CHECK: fcvtas w1, s2                  ; encoding: [0x41,0x00,0x24,0x1e]
; CHECK: fcvtas x1, s2                  ; encoding: [0x41,0x00,0x24,0x9e]
; FP16:  fcvtas w1, h2                  ; encoding: [0x41,0x00,0xe4,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtas  w1, h2
; FP16:  fcvtas x1, h2                  ; encoding: [0x41,0x00,0xe4,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtas  x1, h2

  fcvtau w1, h2
  fcvtau w1, s2
  fcvtau w1, d2
  fcvtau x1, h2
  fcvtau x1, s2
  fcvtau x1, d2

; FP16:  fcvtau w1, h2                  ; encoding: [0x41,0x00,0xe5,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtau  w1, h2
; CHECK: fcvtau w1, s2                  ; encoding: [0x41,0x00,0x25,0x1e]
; CHECK: fcvtau w1, d2                  ; encoding: [0x41,0x00,0x65,0x1e]
; FP16:  fcvtau x1, h2                  ; encoding: [0x41,0x00,0xe5,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtau  x1, h2
; CHECK: fcvtau x1, s2                  ; encoding: [0x41,0x00,0x25,0x9e]
; CHECK: fcvtau x1, d2                  ; encoding: [0x41,0x00,0x65,0x9e]

  fcvtms w1, h2
  fcvtms w1, s2
  fcvtms w1, d2
  fcvtms x1, h2
  fcvtms x1, s2
  fcvtms x1, d2

; FP16:  fcvtms w1, h2                  ; encoding: [0x41,0x00,0xf0,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtms  w1, h2
; CHECK: fcvtms w1, s2                  ; encoding: [0x41,0x00,0x30,0x1e]
; CHECK: fcvtms w1, d2                  ; encoding: [0x41,0x00,0x70,0x1e]
; FP16:  fcvtms x1, h2                  ; encoding: [0x41,0x00,0xf0,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtms  x1, h2
; CHECK: fcvtms x1, s2                  ; encoding: [0x41,0x00,0x30,0x9e]
; CHECK: fcvtms x1, d2                  ; encoding: [0x41,0x00,0x70,0x9e]

  fcvtmu w1, h2
  fcvtmu w1, s2
  fcvtmu w1, d2
  fcvtmu x1, h2
  fcvtmu x1, s2
  fcvtmu x1, d2

; FP16:  fcvtmu w1, h2                  ; encoding: [0x41,0x00,0xf1,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtmu  w1, h2
; CHECK: fcvtmu w1, s2                  ; encoding: [0x41,0x00,0x31,0x1e]
; CHECK: fcvtmu w1, d2                  ; encoding: [0x41,0x00,0x71,0x1e]
; FP16:  fcvtmu x1, h2                  ; encoding: [0x41,0x00,0xf1,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtmu  x1, h2
; CHECK: fcvtmu x1, s2                  ; encoding: [0x41,0x00,0x31,0x9e]
; CHECK: fcvtmu x1, d2                  ; encoding: [0x41,0x00,0x71,0x9e]

  fcvtns w1, h2
  fcvtns w1, s2
  fcvtns w1, d2
  fcvtns x1, h2
  fcvtns x1, s2
  fcvtns x1, d2

; FP16:  fcvtns w1, h2                  ; encoding: [0x41,0x00,0xe0,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtns  w1, h2
; CHECK: fcvtns w1, s2                  ; encoding: [0x41,0x00,0x20,0x1e]
; CHECK: fcvtns w1, d2                  ; encoding: [0x41,0x00,0x60,0x1e]
; FP16:  fcvtns x1, h2                  ; encoding: [0x41,0x00,0xe0,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtns  x1, h2
; CHECK: fcvtns x1, s2                  ; encoding: [0x41,0x00,0x20,0x9e]
; CHECK: fcvtns x1, d2                  ; encoding: [0x41,0x00,0x60,0x9e]

  fcvtnu w1, h2
  fcvtnu w1, s2
  fcvtnu w1, d2
  fcvtnu x1, h2
  fcvtnu x1, s2
  fcvtnu x1, d2

; FP16:  fcvtnu w1, h2                  ; encoding: [0x41,0x00,0xe1,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtnu  w1, h2
; CHECK: fcvtnu w1, s2                  ; encoding: [0x41,0x00,0x21,0x1e]
; CHECK: fcvtnu w1, d2                  ; encoding: [0x41,0x00,0x61,0x1e]
; FP16:  fcvtnu x1, h2                  ; encoding: [0x41,0x00,0xe1,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtnu  x1, h2
; CHECK: fcvtnu x1, s2                  ; encoding: [0x41,0x00,0x21,0x9e]
; CHECK: fcvtnu x1, d2                  ; encoding: [0x41,0x00,0x61,0x9e]

  fcvtps w1, h2
  fcvtps w1, s2
  fcvtps w1, d2
  fcvtps x1, h2
  fcvtps x1, s2
  fcvtps x1, d2

; FP16:  fcvtps w1, h2                  ; encoding: [0x41,0x00,0xe8,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtps  w1, h2
; CHECK: fcvtps w1, s2                  ; encoding: [0x41,0x00,0x28,0x1e]
; CHECK: fcvtps w1, d2                  ; encoding: [0x41,0x00,0x68,0x1e]
; FP16:  fcvtps x1, h2                  ; encoding: [0x41,0x00,0xe8,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtps  x1, h2
; CHECK: fcvtps x1, s2                  ; encoding: [0x41,0x00,0x28,0x9e]
; CHECK: fcvtps x1, d2                  ; encoding: [0x41,0x00,0x68,0x9e]

  fcvtpu w1, h2
  fcvtpu w1, s2
  fcvtpu w1, d2
  fcvtpu x1, h2
  fcvtpu x1, s2
  fcvtpu x1, d2

; FP16:  fcvtpu w1, h2                  ; encoding: [0x41,0x00,0xe9,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtpu  w1, h2
; CHECK: fcvtpu w1, s2                  ; encoding: [0x41,0x00,0x29,0x1e]
; CHECK: fcvtpu w1, d2                  ; encoding: [0x41,0x00,0x69,0x1e]
; FP16:  fcvtpu x1, h2                  ; encoding: [0x41,0x00,0xe9,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtpu  x1, h2
; CHECK: fcvtpu x1, s2                  ; encoding: [0x41,0x00,0x29,0x9e]
; CHECK: fcvtpu x1, d2                  ; encoding: [0x41,0x00,0x69,0x9e]

  fcvtzs w1, h2
  fcvtzs w1, h2, #1
  fcvtzs w1, s2
  fcvtzs w1, s2, #1
  fcvtzs w1, d2
  fcvtzs w1, d2, #1
  fcvtzs x1, h2
  fcvtzs x1, h2, #1
  fcvtzs x1, s2
  fcvtzs x1, s2, #1
  fcvtzs x1, d2
  fcvtzs x1, d2, #1

; FP16:  fcvtzs w1, h2                  ; encoding: [0x41,0x00,0xf8,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtzs  w1, h2
; FP16:  fcvtzs w1, h2, #1              ; encoding: [0x41,0xfc,0xd8,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtzs  w1, h2, #1
; CHECK: fcvtzs w1, s2                  ; encoding: [0x41,0x00,0x38,0x1e]
; CHECK: fcvtzs w1, s2, #1              ; encoding: [0x41,0xfc,0x18,0x1e]
; CHECK: fcvtzs w1, d2                  ; encoding: [0x41,0x00,0x78,0x1e]
; CHECK: fcvtzs w1, d2, #1              ; encoding: [0x41,0xfc,0x58,0x1e]
; FP16:  fcvtzs x1, h2                  ; encoding: [0x41,0x00,0xf8,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtzs  x1, h2
; FP16:  fcvtzs x1, h2, #1              ; encoding: [0x41,0xfc,0xd8,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtzs  x1, h2, #1
; CHECK: fcvtzs x1, s2                  ; encoding: [0x41,0x00,0x38,0x9e]
; CHECK: fcvtzs x1, s2, #1              ; encoding: [0x41,0xfc,0x18,0x9e]
; CHECK: fcvtzs x1, d2                  ; encoding: [0x41,0x00,0x78,0x9e]
; CHECK: fcvtzs x1, d2, #1              ; encoding: [0x41,0xfc,0x58,0x9e]

  fcvtzu w1, h2
  fcvtzu w1, h2, #1
  fcvtzu w1, s2
  fcvtzu w1, s2, #1
  fcvtzu w1, d2
  fcvtzu w1, d2, #1
  fcvtzu x1, h2
  fcvtzu x1, h2, #1
  fcvtzu x1, s2
  fcvtzu x1, s2, #1
  fcvtzu x1, d2
  fcvtzu x1, d2, #1

; FP16:  fcvtzu w1, h2                  ; encoding: [0x41,0x00,0xf9,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtzu  w1, h2
; FP16:  fcvtzu w1, h2, #1              ; encoding: [0x41,0xfc,0xd9,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtzu  w1, h2, #1
; CHECK: fcvtzu w1, s2                  ; encoding: [0x41,0x00,0x39,0x1e]
; CHECK: fcvtzu w1, s2, #1              ; encoding: [0x41,0xfc,0x19,0x1e]
; CHECK: fcvtzu w1, d2                  ; encoding: [0x41,0x00,0x79,0x1e]
; CHECK: fcvtzu w1, d2, #1              ; encoding: [0x41,0xfc,0x59,0x1e]
; FP16:  fcvtzu x1, h2                  ; encoding: [0x41,0x00,0xf9,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtzu  x1, h2
; FP16:  fcvtzu x1, h2, #1              ; encoding: [0x41,0xfc,0xd9,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fcvtzu  x1, h2, #1
; CHECK: fcvtzu x1, s2                  ; encoding: [0x41,0x00,0x39,0x9e]
; CHECK: fcvtzu x1, s2, #1              ; encoding: [0x41,0xfc,0x19,0x9e]
; CHECK: fcvtzu x1, d2                  ; encoding: [0x41,0x00,0x79,0x9e]
; CHECK: fcvtzu x1, d2, #1              ; encoding: [0x41,0xfc,0x59,0x9e]

  scvtf h1, w2
  scvtf h1, w2, #1
  scvtf s1, w2
  scvtf s1, w2, #1
  scvtf d1, w2
  scvtf d1, w2, #1
  scvtf h1, x2
  scvtf h1, x2, #1
  scvtf s1, x2
  scvtf s1, x2, #1
  scvtf d1, x2
  scvtf d1, x2, #1

; FP16:  scvtf  h1, w2                  ; encoding: [0x41,0x00,0xe2,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: scvtf h1, w2
; FP16:  scvtf  h1, w2, #1              ; encoding: [0x41,0xfc,0xc2,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: scvtf h1, w2, #1
; CHECK: scvtf  s1, w2                  ; encoding: [0x41,0x00,0x22,0x1e]
; CHECK: scvtf  s1, w2, #1              ; encoding: [0x41,0xfc,0x02,0x1e]
; CHECK: scvtf  d1, w2                  ; encoding: [0x41,0x00,0x62,0x1e]
; CHECK: scvtf  d1, w2, #1              ; encoding: [0x41,0xfc,0x42,0x1e]
; FP16:  scvtf  h1, x2                  ; encoding: [0x41,0x00,0xe2,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: scvtf h1, x2
; FP16:  scvtf  h1, x2, #1              ; encoding: [0x41,0xfc,0xc2,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: scvtf h1, x2, #1
; CHECK: scvtf  s1, x2                  ; encoding: [0x41,0x00,0x22,0x9e]
; CHECK: scvtf  s1, x2, #1              ; encoding: [0x41,0xfc,0x02,0x9e]
; CHECK: scvtf  d1, x2                  ; encoding: [0x41,0x00,0x62,0x9e]
; CHECK: scvtf  d1, x2, #1              ; encoding: [0x41,0xfc,0x42,0x9e]

  ucvtf h1, w2
  ucvtf h1, w2, #1
  ucvtf s1, w2
  ucvtf s1, w2, #1
  ucvtf d1, w2
  ucvtf d1, w2, #1
  ucvtf h1, x2
  ucvtf h1, x2, #1
  ucvtf s1, x2
  ucvtf s1, x2, #1
  ucvtf d1, x2
  ucvtf d1, x2, #1

; FP16:  ucvtf  h1, w2                  ; encoding: [0x41,0x00,0xe3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: ucvtf h1, w2
; FP16:  ucvtf  h1, w2, #1              ; encoding: [0x41,0xfc,0xc3,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: ucvtf h1, w2, #1
; CHECK: ucvtf  s1, w2                  ; encoding: [0x41,0x00,0x23,0x1e]
; CHECK: ucvtf  s1, w2, #1              ; encoding: [0x41,0xfc,0x03,0x1e]
; CHECK: ucvtf  d1, w2                  ; encoding: [0x41,0x00,0x63,0x1e]
; CHECK: ucvtf  d1, w2, #1              ; encoding: [0x41,0xfc,0x43,0x1e]
; FP16:  ucvtf  h1, x2                  ; encoding: [0x41,0x00,0xe3,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: ucvtf h1, x2
; FP16:  ucvtf  h1, x2, #1              ; encoding: [0x41,0xfc,0xc3,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: ucvtf h1, x2, #1
; CHECK: ucvtf  s1, x2                  ; encoding: [0x41,0x00,0x23,0x9e]
; CHECK: ucvtf  s1, x2, #1              ; encoding: [0x41,0xfc,0x03,0x9e]
; CHECK: ucvtf  d1, x2                  ; encoding: [0x41,0x00,0x63,0x9e]
; CHECK: ucvtf  d1, x2, #1              ; encoding: [0x41,0xfc,0x43,0x9e]

;-----------------------------------------------------------------------------
; Floating-point move
;-----------------------------------------------------------------------------

  fmov h1, w2
  fmov w1, h2
  fmov h1, x2
  fmov x1, h2
  fmov s1, w2
  fmov w1, s2
  fmov d1, x2
  fmov x1, d2

; FP16:  fmov h1, w2                 ; encoding: [0x41,0x00,0xe7,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmov h1, w2
; FP16:  fmov w1, h2                 ; encoding: [0x41,0x00,0xe6,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmov w1, h2
; FP16:  fmov h1, x2                 ; encoding: [0x41,0x00,0xe7,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmov h1, x2
; FP16:  fmov x1, h2                 ; encoding: [0x41,0x00,0xe6,0x9e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmov x1, h2
; CHECK: fmov s1, w2                 ; encoding: [0x41,0x00,0x27,0x1e]
; CHECK: fmov w1, s2                 ; encoding: [0x41,0x00,0x26,0x1e]
; CHECK: fmov d1, x2                 ; encoding: [0x41,0x00,0x67,0x9e]
; CHECK: fmov x1, d2                 ; encoding: [0x41,0x00,0x66,0x9e]

  fmov h1, #0.125
  fmov h1, #0x40
  fmov s1, #0.125
  fmov s1, #0x40
  fmov d1, #0.125
  fmov d1, #0x40
  fmov d1, #-4.843750e-01
  fmov d1, #4.843750e-01
  fmov d3, #3
  fmov h2, #0.0
  fmov s2, #0.0
  fmov d2, #0.0

; FP16:  fmov h1, #0.12500000      ; encoding: [0x01,0x10,0xe8,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmov h1, #0.125
; FP16:  fmov h1, #0.12500000      ; encoding: [0x01,0x10,0xe8,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmov h1, #0x40
; CHECK: fmov s1, #0.12500000      ; encoding: [0x01,0x10,0x28,0x1e]
; CHECK: fmov s1, #0.12500000      ; encoding: [0x01,0x10,0x28,0x1e]
; CHECK: fmov d1, #0.12500000      ; encoding: [0x01,0x10,0x68,0x1e]
; CHECK: fmov d1, #0.12500000      ; encoding: [0x01,0x10,0x68,0x1e]
; CHECK: fmov d1, #-0.48437500     ; encoding: [0x01,0xf0,0x7b,0x1e]
; CHECK: fmov d1, #0.48437500      ; encoding: [0x01,0xf0,0x6b,0x1e]
; CHECK: fmov d3, #3.00000000      ; encoding: [0x03,0x10,0x61,0x1e]
; FP16:  fmov h2, wzr                ; encoding: [0xe2,0x03,0xe7,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmov h2, #0.0
; CHECK: fmov s2, wzr                ; encoding: [0xe2,0x03,0x27,0x1e]
; CHECK: fmov d2, xzr                ; encoding: [0xe2,0x03,0x67,0x9e]

  fmov h1, h2
  fmov s1, s2
  fmov d1, d2

; FP16:  fmov h1, h2                 ; encoding: [0x41,0x40,0xe0,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: fmov h1, h2
; CHECK: fmov s1, s2                 ; encoding: [0x41,0x40,0x20,0x1e]
; CHECK: fmov d1, d2                 ; encoding: [0x41,0x40,0x60,0x1e]


  fmov x2, v5.d[1]
  fmov.d x9, v7[1]
  fmov v1.d[1], x1
  fmov.d v8[1], x6

; CHECK: fmov.d x2, v5[1]               ; encoding: [0xa2,0x00,0xae,0x9e]
; CHECK: fmov.d x9, v7[1]               ; encoding: [0xe9,0x00,0xae,0x9e]
; CHECK: fmov.d v1[1], x1               ; encoding: [0x21,0x00,0xaf,0x9e]
; CHECK: fmov.d v8[1], x6               ; encoding: [0xc8,0x00,0xaf,0x9e]


;-----------------------------------------------------------------------------
; Floating-point round to integral
;-----------------------------------------------------------------------------

  frinta h1, h2
  frinta s1, s2
  frinta d1, d2

; FP16:  frinta h1, h2               ; encoding: [0x41,0x40,0xe6,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: frinta h1, h2
; CHECK: frinta s1, s2               ; encoding: [0x41,0x40,0x26,0x1e]
; CHECK: frinta d1, d2               ; encoding: [0x41,0x40,0x66,0x1e]

  frinti h1, h2
  frinti s1, s2
  frinti d1, d2

; FP16:  frinti h1, h2               ; encoding: [0x41,0xc0,0xe7,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: frinti h1, h2
; CHECK: frinti s1, s2               ; encoding: [0x41,0xc0,0x27,0x1e]
; CHECK: frinti d1, d2               ; encoding: [0x41,0xc0,0x67,0x1e]

  frintm h1, h2
  frintm s1, s2
  frintm d1, d2

; FP16:  frintm h1, h2               ; encoding: [0x41,0x40,0xe5,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: frintm h1, h2
; CHECK: frintm s1, s2               ; encoding: [0x41,0x40,0x25,0x1e]
; CHECK: frintm d1, d2               ; encoding: [0x41,0x40,0x65,0x1e]

  frintn h1, h2
  frintn s1, s2
  frintn d1, d2

; FP16:  frintn h1, h2               ; encoding: [0x41,0x40,0xe4,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: frintn h1, h2
; CHECK: frintn s1, s2               ; encoding: [0x41,0x40,0x24,0x1e]
; CHECK: frintn d1, d2               ; encoding: [0x41,0x40,0x64,0x1e]

  frintp h1, h2
  frintp s1, s2
  frintp d1, d2

; FP16:  frintp h1, h2               ; encoding: [0x41,0xc0,0xe4,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: frintp h1, h2
; CHECK: frintp s1, s2               ; encoding: [0x41,0xc0,0x24,0x1e]
; CHECK: frintp d1, d2               ; encoding: [0x41,0xc0,0x64,0x1e]

  frintx h1, h2
  frintx s1, s2
  frintx d1, d2

; FP16:  frintx h1, h2               ; encoding: [0x41,0x40,0xe7,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: frintx h1, h2
; CHECK: frintx s1, s2               ; encoding: [0x41,0x40,0x27,0x1e]
; CHECK: frintx d1, d2               ; encoding: [0x41,0x40,0x67,0x1e]

  frintz h1, h2
  frintz s1, s2
  frintz d1, d2

; FP16:  frintz h1, h2               ; encoding: [0x41,0xc0,0xe5,0x1e]
; NO-FP16: error: instruction requires:
; NO-FP16-NEXT: frintz h1, h2
; CHECK: frintz s1, s2               ; encoding: [0x41,0xc0,0x25,0x1e]
; CHECK: frintz d1, d2               ; encoding: [0x41,0xc0,0x65,0x1e]

  cmhs d0, d0, d0
  cmtst d0, d0, d0

; CHECK: cmhs d0, d0, d0              ; encoding: [0x00,0x3c,0xe0,0x7e]
; CHECK: cmtst  d0, d0, d0              ; encoding: [0x00,0x8c,0xe0,0x5e]



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
