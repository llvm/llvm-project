; RUN: llvm-mc -triple arm64-apple-darwin -mattr=crypto,fullfp16 -output-asm-variant=1 -show-encoding < %s | FileCheck %s

foo:

  abs.8b  v0, v0
  abs.16b v0, v0
  abs.4h  v0, v0
  abs.8h  v0, v0
  abs.2s  v0, v0
  abs.4s  v0, v0

; CHECK: abs.8b  v0, v0              ; encoding: [0x00,0xb8,0x20,0x0e]
; CHECK: abs.16b v0, v0              ; encoding: [0x00,0xb8,0x20,0x4e]
; CHECK: abs.4h  v0, v0              ; encoding: [0x00,0xb8,0x60,0x0e]
; CHECK: abs.8h  v0, v0              ; encoding: [0x00,0xb8,0x60,0x4e]
; CHECK: abs.2s  v0, v0              ; encoding: [0x00,0xb8,0xa0,0x0e]
; CHECK: abs.4s  v0, v0              ; encoding: [0x00,0xb8,0xa0,0x4e]

  add.8b  v0, v0, v0
  add.16b v0, v0, v0
  add.4h  v0, v0, v0
  add.8h  v0, v0, v0
  add.2s  v0, v0, v0
  add.4s  v0, v0, v0
  add.2d  v0, v0, v0

; CHECK: add.8b  v0, v0, v0          ; encoding: [0x00,0x84,0x20,0x0e]
; CHECK: add.16b v0, v0, v0          ; encoding: [0x00,0x84,0x20,0x4e]
; CHECK: add.4h  v0, v0, v0          ; encoding: [0x00,0x84,0x60,0x0e]
; CHECK: add.8h  v0, v0, v0          ; encoding: [0x00,0x84,0x60,0x4e]
; CHECK: add.2s  v0, v0, v0          ; encoding: [0x00,0x84,0xa0,0x0e]
; CHECK: add.4s  v0, v0, v0          ; encoding: [0x00,0x84,0xa0,0x4e]
; CHECK: add.2d  v0, v0, v0          ; encoding: [0x00,0x84,0xe0,0x4e]

  add d1, d2, d3

; CHECK: add d1, d2, d3              ; encoding: [0x41,0x84,0xe3,0x5e]

  addhn.8b   v0, v0, v0
  addhn2.16b v0, v0, v0
  addhn.4h   v0, v0, v0
  addhn2.8h  v0, v0, v0
  addhn.2s   v0, v0, v0
  addhn2.4s  v0, v0, v0

; CHECK: addhn.8b   v0, v0, v0       ; encoding: [0x00,0x40,0x20,0x0e]
; CHECK: addhn2.16b v0, v0, v0       ; encoding: [0x00,0x40,0x20,0x4e]
; CHECK: addhn.4h   v0, v0, v0       ; encoding: [0x00,0x40,0x60,0x0e]
; CHECK: addhn2.8h  v0, v0, v0       ; encoding: [0x00,0x40,0x60,0x4e]
; CHECK: addhn.2s   v0, v0, v0       ; encoding: [0x00,0x40,0xa0,0x0e]
; CHECK: addhn2.4s  v0, v0, v0       ; encoding: [0x00,0x40,0xa0,0x4e]

  addp.8b  v0, v0, v0
  addp.16b v0, v0, v0
  addp.4h  v0, v0, v0
  addp.8h  v0, v0, v0
  addp.2s  v0, v0, v0
  addp.4s  v0, v0, v0
  addp.2d  v0, v0, v0

; CHECK: addp.8b   v0, v0, v0        ; encoding: [0x00,0xbc,0x20,0x0e]
; CHECK: addp.16b  v0, v0, v0        ; encoding: [0x00,0xbc,0x20,0x4e]
; CHECK: addp.4h   v0, v0, v0        ; encoding: [0x00,0xbc,0x60,0x0e]
; CHECK: addp.8h   v0, v0, v0        ; encoding: [0x00,0xbc,0x60,0x4e]
; CHECK: addp.2s   v0, v0, v0        ; encoding: [0x00,0xbc,0xa0,0x0e]
; CHECK: addp.4s   v0, v0, v0        ; encoding: [0x00,0xbc,0xa0,0x4e]
; CHECK: addp.2d   v0, v0, v0        ; encoding: [0x00,0xbc,0xe0,0x4e]

  addp.2d  d0, v0

; CHECK: addp.2d d0, v0              ; encoding: [0x00,0xb8,0xf1,0x5e]

  addv.8b  b0, v0
  addv.16b b0, v0
  addv.4h  h0, v0
  addv.8h  h0, v0
  addv.4s  s0, v0

; CHECK: addv.8b  b0, v0             ; encoding: [0x00,0xb8,0x31,0x0e]
; CHECK: addv.16b b0, v0             ; encoding: [0x00,0xb8,0x31,0x4e]
; CHECK: addv.4h  h0, v0             ; encoding: [0x00,0xb8,0x71,0x0e]
; CHECK: addv.8h  h0, v0             ; encoding: [0x00,0xb8,0x71,0x4e]
; CHECK: addv.4s  s0, v0             ; encoding: [0x00,0xb8,0xb1,0x4e]


; INS/DUP
  dup.2d  v0, x3
  dup.4s  v0, w3
  dup.2s  v0, w3
  dup.8h  v0, w3
  dup.4h  v0, w3
  dup.16b v0, w3
  dup.8b  v0, w3

  dup v1.2d, x3
  dup v2.4s, w4
  dup v3.2s, w5
  dup v4.8h, w6
  dup v5.4h, w7
  dup v6.16b, w8
  dup v7.8b, w9

; CHECK: dup.2d  v0, x3              ; encoding: [0x60,0x0c,0x08,0x4e]
; CHECK: dup.4s  v0, w3              ; encoding: [0x60,0x0c,0x04,0x4e]
; CHECK: dup.2s  v0, w3              ; encoding: [0x60,0x0c,0x04,0x0e]
; CHECK: dup.8h  v0, w3              ; encoding: [0x60,0x0c,0x02,0x4e]
; CHECK: dup.4h  v0, w3              ; encoding: [0x60,0x0c,0x02,0x0e]
; CHECK: dup.16b v0, w3              ; encoding: [0x60,0x0c,0x01,0x4e]
; CHECK: dup.8b  v0, w3              ; encoding: [0x60,0x0c,0x01,0x0e]

; CHECK: dup.2d	v1, x3               ; encoding: [0x61,0x0c,0x08,0x4e]
; CHECK: dup.4s	v2, w4               ; encoding: [0x82,0x0c,0x04,0x4e]
; CHECK: dup.2s	v3, w5               ; encoding: [0xa3,0x0c,0x04,0x0e]
; CHECK: dup.8h	v4, w6               ; encoding: [0xc4,0x0c,0x02,0x4e]
; CHECK: dup.4h	v5, w7               ; encoding: [0xe5,0x0c,0x02,0x0e]
; CHECK: dup.16b v6, w8              ; encoding: [0x06,0x0d,0x01,0x4e]
; CHECK: dup.8b	v7, w9               ; encoding: [0x27,0x0d,0x01,0x0e]

  dup.2d  v0, v3[1]
  dup.2s  v0, v3[1]
  dup.4s  v0, v3[1]
  dup.4h  v0, v3[1]
  dup.8h  v0, v3[1]
  dup.8b  v0, v3[1]
  dup.16b v0, v3[1]

  dup v7.2d, v9.d[1]
  dup v6.2s, v8.s[1]
  dup v5.4s, v7.s[2]
  dup v4.4h, v6.h[3]
  dup v3.8h, v5.h[4]
  dup v2.8b, v4.b[5]
  dup v1.16b, v3.b[6]

; CHECK: dup.2d  v0, v3[1]           ; encoding: [0x60,0x04,0x18,0x4e]
; CHECK: dup.2s  v0, v3[1]           ; encoding: [0x60,0x04,0x0c,0x0e]
; CHECK: dup.4s  v0, v3[1]           ; encoding: [0x60,0x04,0x0c,0x4e]
; CHECK: dup.4h  v0, v3[1]           ; encoding: [0x60,0x04,0x06,0x0e]
; CHECK: dup.8h  v0, v3[1]           ; encoding: [0x60,0x04,0x06,0x4e]
; CHECK: dup.8b  v0, v3[1]           ; encoding: [0x60,0x04,0x03,0x0e]
; CHECK: dup.16b v0, v3[1]           ; encoding: [0x60,0x04,0x03,0x4e]

; CHECK: dup.2d  v7, v9[1]            ; encoding: [0x27,0x05,0x18,0x4e]
; CHECK: dup.2s  v6, v8[1]            ; encoding: [0x06,0x05,0x0c,0x0e]
; CHECK: dup.4s  v5, v7[2]            ; encoding: [0xe5,0x04,0x14,0x4e]
; CHECK: dup.4h  v4, v6[3]            ; encoding: [0xc4,0x04,0x0e,0x0e]
; CHECK: dup.8h  v3, v5[4]            ; encoding: [0xa3,0x04,0x12,0x4e]
; CHECK: dup.8b  v2, v4[5]            ; encoding: [0x82,0x04,0x0b,0x0e]
; CHECK: dup.16b v1, v3[6]            ; encoding: [0x61,0x04,0x0d,0x4e]

  dup b3, v4[1]
  dup h3, v4[1]
  dup s3, v4[1]
  dup d3, v4[1]
  dup b3, v4.b[1]
  dup h3, v4.h[1]
  dup s3, v4.s[1]
  dup d3, v4.d[1]

  mov b3, v4[1]
  mov h3, v4[1]
  mov s3, v4[1]
  mov d3, v4[1]
  mov b3, v4.b[1]
  mov h3, v4.h[1]
  mov s3, v4.s[1]
  mov d3, v4.d[1]

; CHECK: mov b3, v4[1]               ; encoding: [0x83,0x04,0x03,0x5e]
; CHECK: mov h3, v4[1]               ; encoding: [0x83,0x04,0x06,0x5e]
; CHECK: mov s3, v4[1]               ; encoding: [0x83,0x04,0x0c,0x5e]
; CHECK: mov d3, v4[1]               ; encoding: [0x83,0x04,0x18,0x5e]
; CHECK: mov b3, v4[1]               ; encoding: [0x83,0x04,0x03,0x5e]
; CHECK: mov h3, v4[1]               ; encoding: [0x83,0x04,0x06,0x5e]
; CHECK: mov s3, v4[1]               ; encoding: [0x83,0x04,0x0c,0x5e]
; CHECK: mov d3, v4[1]               ; encoding: [0x83,0x04,0x18,0x5e]

; CHECK: mov b3, v4[1]               ; encoding: [0x83,0x04,0x03,0x5e]
; CHECK: mov h3, v4[1]               ; encoding: [0x83,0x04,0x06,0x5e]
; CHECK: mov s3, v4[1]               ; encoding: [0x83,0x04,0x0c,0x5e]
; CHECK: mov d3, v4[1]               ; encoding: [0x83,0x04,0x18,0x5e]
; CHECK: mov b3, v4[1]               ; encoding: [0x83,0x04,0x03,0x5e]
; CHECK: mov h3, v4[1]               ; encoding: [0x83,0x04,0x06,0x5e]
; CHECK: mov s3, v4[1]               ; encoding: [0x83,0x04,0x0c,0x5e]
; CHECK: mov d3, v4[1]               ; encoding: [0x83,0x04,0x18,0x5e]

  smov.s x3, v2[2]
  smov   x3, v2.s[2]
  umov.s w3, v2[2]
  umov   w3, v2.s[2]
  umov.d x3, v2[1]
  umov   x3, v2.d[1]

; CHECK: smov.s  x3, v2[2]           ; encoding: [0x43,0x2c,0x14,0x4e]
; CHECK: smov.s  x3, v2[2]           ; encoding: [0x43,0x2c,0x14,0x4e]
; CHECK: mov.s  w3, v2[2]           ; encoding: [0x43,0x3c,0x14,0x0e]
; CHECK: mov.s  w3, v2[2]           ; encoding: [0x43,0x3c,0x14,0x0e]
; CHECK: mov.d  x3, v2[1]           ; encoding: [0x43,0x3c,0x18,0x4e]
; CHECK: mov.d  x3, v2[1]           ; encoding: [0x43,0x3c,0x18,0x4e]

  ; MOV aliases for UMOV instructions above

  mov.s w2, v3[3]
  mov   w5, v7.s[2]
  mov.d x11, v13[1]
  mov   x17, v19.d[0]

; CHECK: mov.s  w2, v3[3]               ; encoding: [0x62,0x3c,0x1c,0x0e]
; CHECK: mov.s  w5, v7[2]               ; encoding: [0xe5,0x3c,0x14,0x0e]
; CHECK: mov.d  x11, v13[1]             ; encoding: [0xab,0x3d,0x18,0x4e]
; CHECK: mov.d  x17, v19[0]             ; encoding: [0x71,0x3e,0x08,0x4e]

  ins.d v2[1], x5
  ins.s v2[1], w5
  ins.h v2[1], w5
  ins.b v2[1], w5

  ins   v2.d[1], x5
  ins   v2.s[1], w5
  ins   v2.h[1], w5
  ins   v2.b[1], w5

; CHECK: mov.d v2[1], x5             ; encoding: [0xa2,0x1c,0x18,0x4e]
; CHECK: mov.s v2[1], w5             ; encoding: [0xa2,0x1c,0x0c,0x4e]
; CHECK: mov.h v2[1], w5             ; encoding: [0xa2,0x1c,0x06,0x4e]
; CHECK: mov.b v2[1], w5             ; encoding: [0xa2,0x1c,0x03,0x4e]

; CHECK: mov.d v2[1], x5             ; encoding: [0xa2,0x1c,0x18,0x4e]
; CHECK: mov.s v2[1], w5             ; encoding: [0xa2,0x1c,0x0c,0x4e]
; CHECK: mov.h v2[1], w5             ; encoding: [0xa2,0x1c,0x06,0x4e]
; CHECK: mov.b v2[1], w5             ; encoding: [0xa2,0x1c,0x03,0x4e]

  ins.d v2[1], v15[1]
  ins.s v2[1], v15[1]
  ins.h v2[1], v15[1]
  ins.b v2[1], v15[1]

  ins   v2.d[1], v15.d[0]
  ins   v2.s[3], v15.s[2]
  ins   v2.h[7], v15.h[3]
  ins   v2.b[10], v15.b[5]

; CHECK: mov.d v2[1], v15[1]         ; encoding: [0xe2,0x45,0x18,0x6e]
; CHECK: mov.s v2[1], v15[1]         ; encoding: [0xe2,0x25,0x0c,0x6e]
; CHECK: mov.h v2[1], v15[1]         ; encoding: [0xe2,0x15,0x06,0x6e]
; CHECK: mov.b v2[1], v15[1]         ; encoding: [0xe2,0x0d,0x03,0x6e]

; CHECK: mov.d v2[1], v15[0]         ; encoding: [0xe2,0x05,0x18,0x6e]
; CHECK: mov.s v2[3], v15[2]         ; encoding: [0xe2,0x45,0x1c,0x6e]
; CHECK: mov.h v2[7], v15[3]         ; encoding: [0xe2,0x35,0x1e,0x6e]
; CHECK: mov.b v2[10], v15[5]        ; encoding: [0xe2,0x2d,0x15,0x6e]

; MOV aliases for the above INS instructions.
  mov.d v2[1], x5
  mov.s v3[1], w6
  mov.h v4[1], w7
  mov.b v5[1], w8

  mov   v9.d[1], x2
  mov   v8.s[1], w3
  mov   v7.h[1], w4
  mov   v6.b[1], w5

  mov.d v1[1], v10[1]
  mov.s v2[1], v11[1]
  mov.h v7[1], v12[1]
  mov.b v8[1], v15[1]

  mov   v2.d[1], v15.d[0]
  mov   v7.s[3], v16.s[2]
  mov   v8.h[7], v17.h[3]
  mov   v9.b[10], v18.b[5]

; CHECK: mov.d	v2[1], x5               ; encoding: [0xa2,0x1c,0x18,0x4e]
; CHECK: mov.s	v3[1], w6               ; encoding: [0xc3,0x1c,0x0c,0x4e]
; CHECK: mov.h	v4[1], w7               ; encoding: [0xe4,0x1c,0x06,0x4e]
; CHECK: mov.b	v5[1], w8               ; encoding: [0x05,0x1d,0x03,0x4e]
; CHECK: mov.d	v9[1], x2               ; encoding: [0x49,0x1c,0x18,0x4e]
; CHECK: mov.s	v8[1], w3               ; encoding: [0x68,0x1c,0x0c,0x4e]
; CHECK: mov.h	v7[1], w4               ; encoding: [0x87,0x1c,0x06,0x4e]
; CHECK: mov.b	v6[1], w5               ; encoding: [0xa6,0x1c,0x03,0x4e]
; CHECK: mov.d	v1[1], v10[1]           ; encoding: [0x41,0x45,0x18,0x6e]
; CHECK: mov.s	v2[1], v11[1]           ; encoding: [0x62,0x25,0x0c,0x6e]
; CHECK: mov.h	v7[1], v12[1]           ; encoding: [0x87,0x15,0x06,0x6e]
; CHECK: mov.b	v8[1], v15[1]           ; encoding: [0xe8,0x0d,0x03,0x6e]
; CHECK: mov.d	v2[1], v15[0]           ; encoding: [0xe2,0x05,0x18,0x6e]
; CHECK: mov.s	v7[3], v16[2]           ; encoding: [0x07,0x46,0x1c,0x6e]
; CHECK: mov.h	v8[7], v17[3]           ; encoding: [0x28,0x36,0x1e,0x6e]
; CHECK: mov.b	v9[10], v18[5]          ; encoding: [0x49,0x2e,0x15,0x6e]


  and.8b  v0, v0, v0
  and.16b v0, v0, v0

; CHECK: and.8b  v0, v0, v0          ; encoding: [0x00,0x1c,0x20,0x0e]
; CHECK: and.16b v0, v0, v0          ; encoding: [0x00,0x1c,0x20,0x4e]

  bic.8b  v0, v0, v0

; CHECK: bic.8b  v0, v0, v0          ; encoding: [0x00,0x1c,0x60,0x0e]

  cmeq.8b v0, v0, v0
  cmge.8b v0, v0, v0
  cmgt.8b v0, v0, v0
  cmhi.8b v0, v0, v0
  cmhs.8b v0, v0, v0
  cmtst.8b v0, v0, v0
  fabd.2s v0, v0, v0
  facge.2s  v0, v0, v0
  facgt.2s  v0, v0, v0
  faddp.2s v0, v0, v0
  fadd.2s v0, v0, v0
  fcmeq.2s  v0, v0, v0
  fcmge.2s  v0, v0, v0
  fcmgt.2s  v0, v0, v0
  fdiv.2s v0, v0, v0
  fmaxnmp.2s v0, v0, v0
  fmaxnm.2s v0, v0, v0
  fmaxp.2s v0, v0, v0
  fmax.2s v0, v0, v0
  fminnmp.2s v0, v0, v0
  fminnm.2s v0, v0, v0
  fminp.2s v0, v0, v0
  fmin.2s v0, v0, v0
  fmla.2s v0, v0, v0
  fmls.2s v0, v0, v0
  fmulx.2s v0, v0, v0
  fmul.2s v0, v0, v0
  fmulx	d2, d3, d1
  fmulx	s2, s3, s1
  frecps.2s v0, v0, v0
  frsqrts.2s v0, v0, v0
  fsub.2s v0, v0, v0
  mla.8b v0, v0, v0
  mls.8b v0, v0, v0
  mul.8b v0, v0, v0
  pmul.8b v0, v0, v0
  saba.8b v0, v0, v0
  sabd.8b v0, v0, v0
  shadd.8b v0, v0, v0
  shsub.8b v0, v0, v0
  smaxp.8b v0, v0, v0
  smax.8b v0, v0, v0
  sminp.8b v0, v0, v0
  smin.8b v0, v0, v0
  sqadd.8b v0, v0, v0
  sqdmulh.4h v0, v0, v0
  sqrdmulh.4h v0, v0, v0
  sqrshl.8b v0, v0, v0
  sqshl.8b v0, v0, v0
  sqsub.8b v0, v0, v0
  srhadd.8b v0, v0, v0
  srshl.8b v0, v0, v0
  sshl.8b v0, v0, v0
  sub.8b v0, v0, v0
  uaba.8b v0, v0, v0
  uabd.8b v0, v0, v0
  uhadd.8b v0, v0, v0
  uhsub.8b v0, v0, v0
  umaxp.8b v0, v0, v0
  umax.8b v0, v0, v0
  uminp.8b v0, v0, v0
  umin.8b v0, v0, v0
  uqadd.8b v0, v0, v0
  uqrshl.8b v0, v0, v0
  uqshl.8b v0, v0, v0
  uqsub.8b v0, v0, v0
  urhadd.8b v0, v0, v0
  urshl.8b v0, v0, v0
  ushl.8b v0, v0, v0

; CHECK: cmeq.8b	v0, v0, v0              ; encoding: [0x00,0x8c,0x20,0x2e]
; CHECK: cmge.8b	v0, v0, v0              ; encoding: [0x00,0x3c,0x20,0x0e]
; CHECK: cmgt.8b	v0, v0, v0              ; encoding: [0x00,0x34,0x20,0x0e]
; CHECK: cmhi.8b	v0, v0, v0              ; encoding: [0x00,0x34,0x20,0x2e]
; CHECK: cmhs.8b	v0, v0, v0              ; encoding: [0x00,0x3c,0x20,0x2e]
; CHECK: cmtst.8b	v0, v0, v0      ; encoding: [0x00,0x8c,0x20,0x0e]
; CHECK: fabd.2s	v0, v0, v0              ; encoding: [0x00,0xd4,0xa0,0x2e]
; CHECK: facge.2s	v0, v0, v0      ; encoding: [0x00,0xec,0x20,0x2e]
; CHECK: facgt.2s	v0, v0, v0      ; encoding: [0x00,0xec,0xa0,0x2e]
; CHECK: faddp.2s	v0, v0, v0      ; encoding: [0x00,0xd4,0x20,0x2e]
; CHECK: fadd.2s	v0, v0, v0              ; encoding: [0x00,0xd4,0x20,0x0e]
; CHECK: fcmeq.2s	v0, v0, v0      ; encoding: [0x00,0xe4,0x20,0x0e]
; CHECK: fcmge.2s	v0, v0, v0      ; encoding: [0x00,0xe4,0x20,0x2e]
; CHECK: fcmgt.2s	v0, v0, v0      ; encoding: [0x00,0xe4,0xa0,0x2e]
; CHECK: fdiv.2s	v0, v0, v0              ; encoding: [0x00,0xfc,0x20,0x2e]
; CHECK: fmaxnmp.2s	v0, v0, v0      ; encoding: [0x00,0xc4,0x20,0x2e]
; CHECK: fmaxnm.2s	v0, v0, v0      ; encoding: [0x00,0xc4,0x20,0x0e]
; CHECK: fmaxp.2s	v0, v0, v0      ; encoding: [0x00,0xf4,0x20,0x2e]
; CHECK: fmax.2s	v0, v0, v0              ; encoding: [0x00,0xf4,0x20,0x0e]
; CHECK: fminnmp.2s	v0, v0, v0      ; encoding: [0x00,0xc4,0xa0,0x2e]
; CHECK: fminnm.2s	v0, v0, v0      ; encoding: [0x00,0xc4,0xa0,0x0e]
; CHECK: fminp.2s	v0, v0, v0      ; encoding: [0x00,0xf4,0xa0,0x2e]
; CHECK: fmin.2s	v0, v0, v0              ; encoding: [0x00,0xf4,0xa0,0x0e]
; CHECK: fmla.2s	v0, v0, v0              ; encoding: [0x00,0xcc,0x20,0x0e]
; CHECK: fmls.2s	v0, v0, v0              ; encoding: [0x00,0xcc,0xa0,0x0e]
; CHECK: fmulx.2s	v0, v0, v0      ; encoding: [0x00,0xdc,0x20,0x0e]

; CHECK: fmul.2s	v0, v0, v0              ; encoding: [0x00,0xdc,0x20,0x2e]
; CHECK: fmulx	d2, d3, d1              ; encoding: [0x62,0xdc,0x61,0x5e]
; CHECK: fmulx	s2, s3, s1              ; encoding: [0x62,0xdc,0x21,0x5e]
; CHECK: frecps.2s	v0, v0, v0      ; encoding: [0x00,0xfc,0x20,0x0e]
; CHECK: frsqrts.2s	v0, v0, v0      ; encoding: [0x00,0xfc,0xa0,0x0e]
; CHECK: fsub.2s	v0, v0, v0              ; encoding: [0x00,0xd4,0xa0,0x0e]
; CHECK: mla.8b	v0, v0, v0              ; encoding: [0x00,0x94,0x20,0x0e]
; CHECK: mls.8b	v0, v0, v0              ; encoding: [0x00,0x94,0x20,0x2e]
; CHECK: mul.8b	v0, v0, v0              ; encoding: [0x00,0x9c,0x20,0x0e]
; CHECK: pmul.8b	v0, v0, v0              ; encoding: [0x00,0x9c,0x20,0x2e]
; CHECK: saba.8b	v0, v0, v0              ; encoding: [0x00,0x7c,0x20,0x0e]
; CHECK: sabd.8b	v0, v0, v0              ; encoding: [0x00,0x74,0x20,0x0e]
; CHECK: shadd.8b	v0, v0, v0      ; encoding: [0x00,0x04,0x20,0x0e]
; CHECK: shsub.8b	v0, v0, v0      ; encoding: [0x00,0x24,0x20,0x0e]
; CHECK: smaxp.8b	v0, v0, v0      ; encoding: [0x00,0xa4,0x20,0x0e]
; CHECK: smax.8b	v0, v0, v0              ; encoding: [0x00,0x64,0x20,0x0e]
; CHECK: sminp.8b	v0, v0, v0      ; encoding: [0x00,0xac,0x20,0x0e]
; CHECK: smin.8b	v0, v0, v0              ; encoding: [0x00,0x6c,0x20,0x0e]
; CHECK: sqadd.8b	v0, v0, v0      ; encoding: [0x00,0x0c,0x20,0x0e]
; CHECK: sqdmulh.4h v0, v0, v0 ; encoding: [0x00,0xb4,0x60,0x0e]
; CHECK: sqrdmulh.4h v0, v0, v0 ; encoding: [0x00,0xb4,0x60,0x2e]
; CHECK: sqrshl.8b	v0, v0, v0      ; encoding: [0x00,0x5c,0x20,0x0e]
; CHECK: sqshl.8b	v0, v0, v0      ; encoding: [0x00,0x4c,0x20,0x0e]
; CHECK: sqsub.8b	v0, v0, v0      ; encoding: [0x00,0x2c,0x20,0x0e]
; CHECK: srhadd.8b	v0, v0, v0      ; encoding: [0x00,0x14,0x20,0x0e]
; CHECK: srshl.8b	v0, v0, v0      ; encoding: [0x00,0x54,0x20,0x0e]
; CHECK: sshl.8b	v0, v0, v0              ; encoding: [0x00,0x44,0x20,0x0e]
; CHECK: sub.8b	v0, v0, v0              ; encoding: [0x00,0x84,0x20,0x2e]
; CHECK: uaba.8b	v0, v0, v0              ; encoding: [0x00,0x7c,0x20,0x2e]
; CHECK: uabd.8b	v0, v0, v0              ; encoding: [0x00,0x74,0x20,0x2e]
; CHECK: uhadd.8b	v0, v0, v0      ; encoding: [0x00,0x04,0x20,0x2e]
; CHECK: uhsub.8b	v0, v0, v0      ; encoding: [0x00,0x24,0x20,0x2e]
; CHECK: umaxp.8b	v0, v0, v0      ; encoding: [0x00,0xa4,0x20,0x2e]
; CHECK: umax.8b	v0, v0, v0              ; encoding: [0x00,0x64,0x20,0x2e]
; CHECK: uminp.8b	v0, v0, v0      ; encoding: [0x00,0xac,0x20,0x2e]
; CHECK: umin.8b	v0, v0, v0              ; encoding: [0x00,0x6c,0x20,0x2e]
; CHECK: uqadd.8b	v0, v0, v0      ; encoding: [0x00,0x0c,0x20,0x2e]
; CHECK: uqrshl.8b	v0, v0, v0      ; encoding: [0x00,0x5c,0x20,0x2e]
; CHECK: uqshl.8b	v0, v0, v0      ; encoding: [0x00,0x4c,0x20,0x2e]
; CHECK: uqsub.8b	v0, v0, v0      ; encoding: [0x00,0x2c,0x20,0x2e]
; CHECK: urhadd.8b	v0, v0, v0      ; encoding: [0x00,0x14,0x20,0x2e]
; CHECK: urshl.8b	v0, v0, v0      ; encoding: [0x00,0x54,0x20,0x2e]
; CHECK: ushl.8b	v0, v0, v0              ; encoding: [0x00,0x44,0x20,0x2e]

  fabd.4h v0, v0, v0
  facge.4h  v0, v0, v0
  facgt.4h  v0, v0, v0
  faddp.4h v0, v0, v0
  fadd.4h v0, v0, v0
  fcmeq.4h  v0, v0, v0
  fcmge.4h  v0, v0, v0
  fcmgt.4h  v0, v0, v0
  fdiv.4h v0, v0, v0
  fmaxnmp.4h v0, v0, v0
  fmaxnm.4h v0, v0, v0
  fmaxp.4h v0, v0, v0
  fmax.4h v0, v0, v0
  fminnmp.4h v0, v0, v0
  fminnm.4h v0, v0, v0
  fminp.4h v0, v0, v0
  fmin.4h v0, v0, v0
  fmla.4h v0, v0, v0
  fmls.4h v0, v0, v0
  fmulx.4h v0, v0, v0
  fmul.4h v0, v0, v0
  frecps.4h v0, v0, v0
  frsqrts.4h v0, v0, v0
  fsub.4h v0, v0, v0

; CHECK: fabd.4h    v0, v0, v0     ; encoding: [0x00,0x14,0xc0,0x2e]
; CHECK: facge.4h   v0, v0, v0     ; encoding: [0x00,0x2c,0x40,0x2e]
; CHECK: facgt.4h   v0, v0, v0     ; encoding: [0x00,0x2c,0xc0,0x2e]
; CHECK: faddp.4h   v0, v0, v0     ; encoding: [0x00,0x14,0x40,0x2e]
; CHECK: fadd.4h    v0, v0, v0     ; encoding: [0x00,0x14,0x40,0x0e]
; CHECK: fcmeq.4h   v0, v0, v0     ; encoding: [0x00,0x24,0x40,0x0e]
; CHECK: fcmge.4h   v0, v0, v0     ; encoding: [0x00,0x24,0x40,0x2e]
; CHECK: fcmgt.4h   v0, v0, v0     ; encoding: [0x00,0x24,0xc0,0x2e]
; CHECK: fdiv.4h    v0, v0, v0     ; encoding: [0x00,0x3c,0x40,0x2e]
; CHECK: fmaxnmp.4h v0, v0, v0     ; encoding: [0x00,0x04,0x40,0x2e]
; CHECK: fmaxnm.4h  v0, v0, v0     ; encoding: [0x00,0x04,0x40,0x0e]
; CHECK: fmaxp.4h   v0, v0, v0     ; encoding: [0x00,0x34,0x40,0x2e]
; CHECK: fmax.4h    v0, v0, v0     ; encoding: [0x00,0x34,0x40,0x0e]
; CHECK: fminnmp.4h v0, v0, v0     ; encoding: [0x00,0x04,0xc0,0x2e]
; CHECK: fminnm.4h  v0, v0, v0     ; encoding: [0x00,0x04,0xc0,0x0e]
; CHECK: fminp.4h   v0, v0, v0     ; encoding: [0x00,0x34,0xc0,0x2e]
; CHECK: fmin.4h    v0, v0, v0     ; encoding: [0x00,0x34,0xc0,0x0e]
; CHECK: fmla.4h    v0, v0, v0     ; encoding: [0x00,0x0c,0x40,0x0e]
; CHECK: fmls.4h    v0, v0, v0     ; encoding: [0x00,0x0c,0xc0,0x0e]
; CHECK: fmulx.4h   v0, v0, v0     ; encoding: [0x00,0x1c,0x40,0x0e]
; CHECK: fmul.4h    v0, v0, v0     ; encoding: [0x00,0x1c,0x40,0x2e]
; CHECK: frecps.4h  v0, v0, v0     ; encoding: [0x00,0x3c,0x40,0x0e]
; CHECK: frsqrts.4h v0, v0, v0     ; encoding: [0x00,0x3c,0xc0,0x0e]
; CHECK: fsub.4h    v0, v0, v0     ; encoding: [0x00,0x14,0xc0,0x0e]

  fabd.8h v0, v0, v0
  facge.8h  v0, v0, v0
  facgt.8h  v0, v0, v0
  faddp.8h v0, v0, v0
  fadd.8h v0, v0, v0
  fcmeq.8h  v0, v0, v0
  fcmge.8h  v0, v0, v0
  fcmgt.8h  v0, v0, v0
  fdiv.8h v0, v0, v0
  fmaxnmp.8h v0, v0, v0
  fmaxnm.8h v0, v0, v0
  fmaxp.8h v0, v0, v0
  fmax.8h v0, v0, v0
  fminnmp.8h v0, v0, v0
  fminnm.8h v0, v0, v0
  fminp.8h v0, v0, v0
  fmin.8h v0, v0, v0
  fmla.8h v0, v0, v0
  fmls.8h v0, v0, v0
  fmulx.8h v0, v0, v0
  fmul.8h v0, v0, v0
  frecps.8h v0, v0, v0
  frsqrts.8h v0, v0, v0
  fsub.8h v0, v0, v0

; CHECK: fabd.8h v0, v0, v0              ; encoding: [0x00,0x14,0xc0,0x6e]
; CHECK: facge.8h        v0, v0, v0      ; encoding: [0x00,0x2c,0x40,0x6e]
; CHECK: facgt.8h        v0, v0, v0      ; encoding: [0x00,0x2c,0xc0,0x6e]
; CHECK: faddp.8h        v0, v0, v0      ; encoding: [0x00,0x14,0x40,0x6e]
; CHECK: fadd.8h v0, v0, v0              ; encoding: [0x00,0x14,0x40,0x4e]
; CHECK: fcmeq.8h        v0, v0, v0      ; encoding: [0x00,0x24,0x40,0x4e]
; CHECK: fcmge.8h        v0, v0, v0      ; encoding: [0x00,0x24,0x40,0x6e]
; CHECK: fcmgt.8h        v0, v0, v0      ; encoding: [0x00,0x24,0xc0,0x6e]
; CHECK: fdiv.8h v0, v0, v0              ; encoding: [0x00,0x3c,0x40,0x6e]
; CHECK: fmaxnmp.8h      v0, v0, v0      ; encoding: [0x00,0x04,0x40,0x6e]
; CHECK: fmaxnm.8h       v0, v0, v0      ; encoding: [0x00,0x04,0x40,0x4e]
; CHECK: fmaxp.8h        v0, v0, v0      ; encoding: [0x00,0x34,0x40,0x6e]
; CHECK: fmax.8h v0, v0, v0              ; encoding: [0x00,0x34,0x40,0x4e]
; CHECK: fminnmp.8h      v0, v0, v0      ; encoding: [0x00,0x04,0xc0,0x6e]
; CHECK: fminnm.8h       v0, v0, v0      ; encoding: [0x00,0x04,0xc0,0x4e]
; CHECK: fminp.8h        v0, v0, v0      ; encoding: [0x00,0x34,0xc0,0x6e]
; CHECK: fmin.8h v0, v0, v0              ; encoding: [0x00,0x34,0xc0,0x4e]
; CHECK: fmla.8h v0, v0, v0              ; encoding: [0x00,0x0c,0x40,0x4e]
; CHECK: fmls.8h v0, v0, v0              ; encoding: [0x00,0x0c,0xc0,0x4e]
; CHECK: fmulx.8h        v0, v0, v0      ; encoding: [0x00,0x1c,0x40,0x4e]
; CHECK: fmul.8h v0, v0, v0              ; encoding: [0x00,0x1c,0x40,0x6e]
; CHECK: frecps.8h       v0, v0, v0      ; encoding: [0x00,0x3c,0x40,0x4e]
; CHECK: frsqrts.8h      v0, v0, v0      ; encoding: [0x00,0x3c,0xc0,0x4e]
; CHECK: fsub.8h v0, v0, v0              ; encoding: [0x00,0x14,0xc0,0x4e]

  bif.8b v0, v0, v0
  bit.8b v0, v0, v0
  bsl.8b v0, v0, v0
  eor.8b v0, v0, v0
  orn.8b v0, v0, v0
  orr.8b v0, v0, v1

; CHECK: bif.8b	v0, v0, v0              ; encoding: [0x00,0x1c,0xe0,0x2e]
; CHECK: bit.8b	v0, v0, v0              ; encoding: [0x00,0x1c,0xa0,0x2e]
; CHECK: bsl.8b	v0, v0, v0              ; encoding: [0x00,0x1c,0x60,0x2e]
; CHECK: eor.8b	v0, v0, v0              ; encoding: [0x00,0x1c,0x20,0x2e]
; CHECK: orn.8b	v0, v0, v0              ; encoding: [0x00,0x1c,0xe0,0x0e]
; CHECK: orr.8b v0, v0, v1              ; encoding: [0x00,0x1c,0xa1,0x0e]

  sadalp.4h   v0, v0
  sadalp.8h  v0, v0
  sadalp.2s   v0, v0
  sadalp.4s   v0, v0
  sadalp.1d   v0, v0
  sadalp.2d   v0, v0

; CHECK: sadalp.4h	v0, v0          ; encoding: [0x00,0x68,0x20,0x0e]
; CHECK: sadalp.8h	v0, v0          ; encoding: [0x00,0x68,0x20,0x4e]
; CHECK: sadalp.2s	v0, v0          ; encoding: [0x00,0x68,0x60,0x0e]
; CHECK: sadalp.4s	v0, v0          ; encoding: [0x00,0x68,0x60,0x4e]
; CHECK: sadalp.1d	v0, v0          ; encoding: [0x00,0x68,0xa0,0x0e]
; CHECK: sadalp.2d	v0, v0          ; encoding: [0x00,0x68,0xa0,0x4e]

  cls.8b      v0, v0
  clz.8b      v0, v0
  cnt.8b      v0, v0
  fabs.2s     v0, v0
  fneg.2s     v0, v0
  frecpe.2s   v0, v0
  frinta.2s   v0, v0
  frintx.2s   v0, v0
  frinti.2s   v0, v0
  frintm.2s   v0, v0
  frintn.2s   v0, v0
  frintp.2s   v0, v0
  frintz.2s   v0, v0
  frsqrte.2s  v0, v0
  fsqrt.2s    v0, v0
  neg.8b      v0, v0
  not.8b      v0, v0
  rbit.8b     v0, v0
  rev16.8b    v0, v0
  rev32.8b    v0, v0
  rev64.8b    v0, v0
  sadalp.4h   v0, v0
  saddlp.4h	  v0, v0
  scvtf.2s    v0, v0
  sqabs.8b    v0, v0
  sqneg.8b    v0, v0
  sqxtn.8b    v0, v0
  sqxtun.8b   v0, v0
  suqadd.8b   v0, v0
  uadalp.4h   v0, v0
  uaddlp.4h   v0, v0
  ucvtf.2s    v0, v0
  uqxtn.8b    v0, v0
  urecpe.2s   v0, v0
  ursqrte.2s  v0, v0
  usqadd.8b   v0, v0
  xtn.8b      v0, v0
  shll.8h v1, v2, #8
  shll.4s v3, v4, #16
  shll.2d v5, v6, #32
  shll2.8h v7, v8, #8
  shll2.4s v9, v10, #16
  shll2.2d v11, v12, #32
  shll v1.8h, v2.8b, #8
  shll v1.4s, v2.4h, #16
  shll v1.2d, v2.2s, #32
  shll2 v1.8h, v2.16b, #8
  shll2 v1.4s, v2.8h, #16
  shll2 v1.2d, v2.4s, #32

; CHECK: cls.8b	v0, v0                  ; encoding: [0x00,0x48,0x20,0x0e]
; CHECK: clz.8b	v0, v0                  ; encoding: [0x00,0x48,0x20,0x2e]
; CHECK: cnt.8b	v0, v0                  ; encoding: [0x00,0x58,0x20,0x0e]
; CHECK: fabs.2s	v0, v0                  ; encoding: [0x00,0xf8,0xa0,0x0e]
; CHECK: fneg.2s	v0, v0                  ; encoding: [0x00,0xf8,0xa0,0x2e]
; CHECK: frecpe.2s	v0, v0          ; encoding: [0x00,0xd8,0xa1,0x0e]
; CHECK: frinta.2s	v0, v0          ; encoding: [0x00,0x88,0x21,0x2e]
; CHECK: frintx.2s	v0, v0          ; encoding: [0x00,0x98,0x21,0x2e]
; CHECK: frinti.2s	v0, v0          ; encoding: [0x00,0x98,0xa1,0x2e]
; CHECK: frintm.2s	v0, v0          ; encoding: [0x00,0x98,0x21,0x0e]
; CHECK: frintn.2s	v0, v0          ; encoding: [0x00,0x88,0x21,0x0e]
; CHECK: frintp.2s	v0, v0          ; encoding: [0x00,0x88,0xa1,0x0e]
; CHECK: frintz.2s	v0, v0          ; encoding: [0x00,0x98,0xa1,0x0e]
; CHECK: frsqrte.2s	v0, v0          ; encoding: [0x00,0xd8,0xa1,0x2e]
; CHECK: fsqrt.2s	v0, v0          ; encoding: [0x00,0xf8,0xa1,0x2e]
; CHECK: neg.8b	v0, v0                  ; encoding: [0x00,0xb8,0x20,0x2e]
; CHECK: mvn.8b	v0, v0                  ; encoding: [0x00,0x58,0x20,0x2e]
; CHECK: rbit.8b	v0, v0                  ; encoding: [0x00,0x58,0x60,0x2e]
; CHECK: rev16.8b	v0, v0          ; encoding: [0x00,0x18,0x20,0x0e]
; CHECK: rev32.8b	v0, v0          ; encoding: [0x00,0x08,0x20,0x2e]
; CHECK: rev64.8b	v0, v0          ; encoding: [0x00,0x08,0x20,0x0e]
; CHECK: sadalp.4h	v0, v0          ; encoding: [0x00,0x68,0x20,0x0e]
; CHECK: saddlp.4h	v0, v0          ; encoding: [0x00,0x28,0x20,0x0e]
; CHECK: scvtf.2s	v0, v0          ; encoding: [0x00,0xd8,0x21,0x0e]
; CHECK: sqabs.8b	v0, v0          ; encoding: [0x00,0x78,0x20,0x0e]
; CHECK: sqneg.8b	v0, v0          ; encoding: [0x00,0x78,0x20,0x2e]
; CHECK: sqxtn.8b	v0, v0          ; encoding: [0x00,0x48,0x21,0x0e]
; CHECK: sqxtun.8b	v0, v0          ; encoding: [0x00,0x28,0x21,0x2e]
; CHECK: suqadd.8b	v0, v0          ; encoding: [0x00,0x38,0x20,0x0e]
; CHECK: uadalp.4h	v0, v0          ; encoding: [0x00,0x68,0x20,0x2e]
; CHECK: uaddlp.4h	v0, v0          ; encoding: [0x00,0x28,0x20,0x2e]
; CHECK: ucvtf.2s	v0, v0          ; encoding: [0x00,0xd8,0x21,0x2e]
; CHECK: uqxtn.8b	v0, v0          ; encoding: [0x00,0x48,0x21,0x2e]
; CHECK: urecpe.2s	v0, v0          ; encoding: [0x00,0xc8,0xa1,0x0e]
; CHECK: ursqrte.2s	v0, v0          ; encoding: [0x00,0xc8,0xa1,0x2e]
; CHECK: usqadd.8b	v0, v0          ; encoding: [0x00,0x38,0x20,0x2e]
; CHECK: xtn.8b	v0, v0                  ; encoding: [0x00,0x28,0x21,0x0e]
; CHECK: shll.8h	v1, v2, #8      ; encoding: [0x41,0x38,0x21,0x2e]
; CHECK: shll.4s	v3, v4, #16     ; encoding: [0x83,0x38,0x61,0x2e]
; CHECK: shll.2d	v5, v6, #32     ; encoding: [0xc5,0x38,0xa1,0x2e]
; CHECK: shll2.8h	v7, v8, #8      ; encoding: [0x07,0x39,0x21,0x6e]
; CHECK: shll2.4s	v9, v10, #16    ; encoding: [0x49,0x39,0x61,0x6e]
; CHECK: shll2.2d	v11, v12, #32   ; encoding: [0x8b,0x39,0xa1,0x6e]
; CHECK: shll.8h	v1, v2, #8      ; encoding: [0x41,0x38,0x21,0x2e]
; CHECK: shll.4s	v1, v2, #16     ; encoding: [0x41,0x38,0x61,0x2e]
; CHECK: shll.2d	v1, v2, #32     ; encoding: [0x41,0x38,0xa1,0x2e]
; CHECK: shll2.8h	v1, v2, #8      ; encoding: [0x41,0x38,0x21,0x6e]
; CHECK: shll2.4s	v1, v2, #16     ; encoding: [0x41,0x38,0x61,0x6e]
; CHECK: shll2.2d	v1, v2, #32     ; encoding: [0x41,0x38,0xa1,0x6e]

  fabs.4h     v0, v0
  fneg.4h     v0, v0
  frecpe.4h   v0, v0
  frinta.4h   v0, v0
  frintx.4h   v0, v0
  frinti.4h   v0, v0
  frintm.4h   v0, v0
  frintn.4h   v0, v0
  frintp.4h   v0, v0
  frintz.4h   v0, v0
  frsqrte.4h  v0, v0
  fsqrt.4h    v0, v0

; CHECK: fabs.4h v0, v0                  ; encoding: [0x00,0xf8,0xf8,0x0e]
; CHECK: fneg.4h v0, v0                  ; encoding: [0x00,0xf8,0xf8,0x2e]
; CHECK: frecpe.4h       v0, v0          ; encoding: [0x00,0xd8,0xf9,0x0e]
; CHECK: frinta.4h       v0, v0          ; encoding: [0x00,0x88,0x79,0x2e]
; CHECK: frintx.4h       v0, v0          ; encoding: [0x00,0x98,0x79,0x2e]
; CHECK: frinti.4h       v0, v0          ; encoding: [0x00,0x98,0xf9,0x2e]
; CHECK: frintm.4h       v0, v0          ; encoding: [0x00,0x98,0x79,0x0e]
; CHECK: frintn.4h       v0, v0          ; encoding: [0x00,0x88,0x79,0x0e]
; CHECK: frintp.4h       v0, v0          ; encoding: [0x00,0x88,0xf9,0x0e]
; CHECK: frintz.4h       v0, v0          ; encoding: [0x00,0x98,0xf9,0x0e]
; CHECK: frsqrte.4h      v0, v0          ; encoding: [0x00,0xd8,0xf9,0x2e]
; CHECK: fsqrt.4h        v0, v0          ; encoding: [0x00,0xf8,0xf9,0x2e]

  fabs.8h     v0, v0
  fneg.8h     v0, v0
  frecpe.8h   v0, v0
  frinta.8h   v0, v0
  frintx.8h   v0, v0
  frinti.8h   v0, v0
  frintm.8h   v0, v0
  frintn.8h   v0, v0
  frintp.8h   v0, v0
  frintz.8h   v0, v0
  frsqrte.8h  v0, v0
  fsqrt.8h    v0, v0

; CHECK: fabs.8h v0, v0                  ; encoding: [0x00,0xf8,0xf8,0x4e]
; CHECK: fneg.8h v0, v0                  ; encoding: [0x00,0xf8,0xf8,0x6e]
; CHECK: frecpe.8h       v0, v0          ; encoding: [0x00,0xd8,0xf9,0x4e]
; CHECK: frinta.8h       v0, v0          ; encoding: [0x00,0x88,0x79,0x6e]
; CHECK: frintx.8h       v0, v0          ; encoding: [0x00,0x98,0x79,0x6e]
; CHECK: frinti.8h       v0, v0          ; encoding: [0x00,0x98,0xf9,0x6e]
; CHECK: frintm.8h       v0, v0          ; encoding: [0x00,0x98,0x79,0x4e]
; CHECK: frintn.8h       v0, v0          ; encoding: [0x00,0x88,0x79,0x4e]
; CHECK: frintp.8h       v0, v0          ; encoding: [0x00,0x88,0xf9,0x4e]
; CHECK: frintz.8h       v0, v0          ; encoding: [0x00,0x98,0xf9,0x4e]
; CHECK: frsqrte.8h      v0, v0          ; encoding: [0x00,0xd8,0xf9,0x6e]
; CHECK: fsqrt.8h        v0, v0          ; encoding: [0x00,0xf8,0xf9,0x6e]

  cmeq.8b   v0, v0, #0
  cmeq.16b  v0, v0, #0
  cmeq.4h   v0, v0, #0
  cmeq.8h   v0, v0, #0
  cmeq.2s   v0, v0, #0
  cmeq.4s   v0, v0, #0
  cmeq.2d   v0, v0, #0

; CHECK: cmeq.8b	v0, v0, #0              ; encoding: [0x00,0x98,0x20,0x0e]
; CHECK: cmeq.16b	v0, v0, #0      ; encoding: [0x00,0x98,0x20,0x4e]
; CHECK: cmeq.4h	v0, v0, #0              ; encoding: [0x00,0x98,0x60,0x0e]
; CHECK: cmeq.8h	v0, v0, #0              ; encoding: [0x00,0x98,0x60,0x4e]
; CHECK: cmeq.2s	v0, v0, #0              ; encoding: [0x00,0x98,0xa0,0x0e]
; CHECK: cmeq.4s	v0, v0, #0              ; encoding: [0x00,0x98,0xa0,0x4e]
; CHECK: cmeq.2d	v0, v0, #0              ; encoding: [0x00,0x98,0xe0,0x4e]

  cmge.8b   v0, v0, #0
  cmgt.8b   v0, v0, #0
  cmle.8b   v0, v0, #0
  cmlt.8b   v0, v0, #0
  fcmeq.2s  v0, v0, #0
  fcmge.2s  v0, v0, #0
  fcmgt.2s  v0, v0, #0
  fcmle.2s  v0, v0, #0
  fcmlt.2s  v0, v0, #0

; ARM verbose mode aliases
  cmlt v8.8b, v14.8b, #0
  cmlt v8.16b, v14.16b, #0
  cmlt v8.4h, v14.4h, #0
  cmlt v8.8h, v14.8h, #0
  cmlt v8.2s, v14.2s, #0
  cmlt v8.4s, v14.4s, #0
  cmlt v8.2d, v14.2d, #0

; CHECK: cmge.8b	v0, v0, #0              ; encoding: [0x00,0x88,0x20,0x2e]
; CHECK: cmgt.8b	v0, v0, #0              ; encoding: [0x00,0x88,0x20,0x0e]
; CHECK: cmle.8b	v0, v0, #0              ; encoding: [0x00,0x98,0x20,0x2e]
; CHECK: cmlt.8b	v0, v0, #0              ; encoding: [0x00,0xa8,0x20,0x0e]
; CHECK: fcmeq.2s	v0, v0, #0.0      ; encoding: [0x00,0xd8,0xa0,0x0e]
; CHECK: fcmge.2s	v0, v0, #0.0      ; encoding: [0x00,0xc8,0xa0,0x2e]
; CHECK: fcmgt.2s	v0, v0, #0.0      ; encoding: [0x00,0xc8,0xa0,0x0e]
; CHECK: fcmle.2s	v0, v0, #0.0      ; encoding: [0x00,0xd8,0xa0,0x2e]
; CHECK: fcmlt.2s	v0, v0, #0.0      ; encoding: [0x00,0xe8,0xa0,0x0e]
; CHECK: cmlt.8b	v8, v14, #0             ; encoding: [0xc8,0xa9,0x20,0x0e]
; CHECK: cmlt.16b	v8, v14, #0     ; encoding: [0xc8,0xa9,0x20,0x4e]
; CHECK: cmlt.4h	v8, v14, #0             ; encoding: [0xc8,0xa9,0x60,0x0e]
; CHECK: cmlt.8h	v8, v14, #0             ; encoding: [0xc8,0xa9,0x60,0x4e]
; CHECK: cmlt.2s	v8, v14, #0             ; encoding: [0xc8,0xa9,0xa0,0x0e]
; CHECK: cmlt.4s	v8, v14, #0             ; encoding: [0xc8,0xa9,0xa0,0x4e]
; CHECK: cmlt.2d	v8, v14, #0             ; encoding: [0xc8,0xa9,0xe0,0x4e]


;===-------------------------------------------------------------------------===
; AdvSIMD Floating-point <-> Integer Conversions
;===-------------------------------------------------------------------------===

  fcvtas.2s   v0, v0
  fcvtas.4s   v0, v0
  fcvtas.2d   v0, v0
  fcvtas      s0, s0
  fcvtas      d0, d0

; CHECK: fcvtas.2s  v0, v0           ; encoding: [0x00,0xc8,0x21,0x0e]
; CHECK: fcvtas.4s  v0, v0           ; encoding: [0x00,0xc8,0x21,0x4e]
; CHECK: fcvtas.2d  v0, v0           ; encoding: [0x00,0xc8,0x61,0x4e]
; CHECK: fcvtas     s0, s0           ; encoding: [0x00,0xc8,0x21,0x5e]
; CHECK: fcvtas     d0, d0           ; encoding: [0x00,0xc8,0x61,0x5e]

  fcvtau.2s   v0, v0
  fcvtau.4s   v0, v0
  fcvtau.2d   v0, v0
  fcvtau      s0, s0
  fcvtau      d0, d0

; CHECK: fcvtau.2s  v0, v0           ; encoding: [0x00,0xc8,0x21,0x2e]
; CHECK: fcvtau.4s  v0, v0           ; encoding: [0x00,0xc8,0x21,0x6e]
; CHECK: fcvtau.2d  v0, v0           ; encoding: [0x00,0xc8,0x61,0x6e]
; CHECK: fcvtau     s0, s0           ; encoding: [0x00,0xc8,0x21,0x7e]
; CHECK: fcvtau     d0, d0           ; encoding: [0x00,0xc8,0x61,0x7e]

  fcvtl   v1.4s, v5.4h
  fcvtl   v2.2d, v6.2s
  fcvtl2  v3.4s, v7.8h
  fcvtl2  v4.2d, v8.4s

; CHECK: fcvtl	v1.4s, v5.4h            ; encoding: [0xa1,0x78,0x21,0x0e]
; CHECK: fcvtl	v2.2d, v6.2s            ; encoding: [0xc2,0x78,0x61,0x0e]
; CHECK: fcvtl2	v3.4s, v7.8h            ; encoding: [0xe3,0x78,0x21,0x4e]
; CHECK: fcvtl2	v4.2d, v8.4s            ; encoding: [0x04,0x79,0x61,0x4e]

  fcvtms.2s  v0, v0
  fcvtms.4s  v0, v0
  fcvtms.2d  v0, v0
  fcvtms     s0, s0
  fcvtms     d0, d0

; CHECK: fcvtms.2s v0, v0            ; encoding: [0x00,0xb8,0x21,0x0e]
; CHECK: fcvtms.4s v0, v0            ; encoding: [0x00,0xb8,0x21,0x4e]
; CHECK: fcvtms.2d v0, v0            ; encoding: [0x00,0xb8,0x61,0x4e]
; CHECK: fcvtms    s0, s0            ; encoding: [0x00,0xb8,0x21,0x5e]
; CHECK: fcvtms    d0, d0            ; encoding: [0x00,0xb8,0x61,0x5e]

  fcvtmu.2s   v0, v0
  fcvtmu.4s   v0, v0
  fcvtmu.2d   v0, v0
  fcvtmu      s0, s0
  fcvtmu      d0, d0

; CHECK: fcvtmu.2s v0, v0            ; encoding: [0x00,0xb8,0x21,0x2e]
; CHECK: fcvtmu.4s v0, v0            ; encoding: [0x00,0xb8,0x21,0x6e]
; CHECK: fcvtmu.2d v0, v0            ; encoding: [0x00,0xb8,0x61,0x6e]
; CHECK: fcvtmu    s0, s0            ; encoding: [0x00,0xb8,0x21,0x7e]
; CHECK: fcvtmu    d0, d0            ; encoding: [0x00,0xb8,0x61,0x7e]

  fcvtns.2s   v0, v0
  fcvtns.4s   v0, v0
  fcvtns.2d   v0, v0
  fcvtns      s0, s0
  fcvtns      d0, d0

; CHECK: fcvtns.2s v0, v0            ; encoding: [0x00,0xa8,0x21,0x0e]
; CHECK: fcvtns.4s v0, v0            ; encoding: [0x00,0xa8,0x21,0x4e]
; CHECK: fcvtns.2d v0, v0            ; encoding: [0x00,0xa8,0x61,0x4e]
; CHECK: fcvtns    s0, s0            ; encoding: [0x00,0xa8,0x21,0x5e]
; CHECK: fcvtns    d0, d0            ; encoding: [0x00,0xa8,0x61,0x5e]

  fcvtnu.2s   v0, v0
  fcvtnu.4s   v0, v0
  fcvtnu.2d   v0, v0
  fcvtnu      s0, s0
  fcvtnu      d0, d0

; CHECK: fcvtnu.2s v0, v0            ; encoding: [0x00,0xa8,0x21,0x2e]
; CHECK: fcvtnu.4s v0, v0            ; encoding: [0x00,0xa8,0x21,0x6e]
; CHECK: fcvtnu.2d v0, v0            ; encoding: [0x00,0xa8,0x61,0x6e]
; CHECK: fcvtnu    s0, s0            ; encoding: [0x00,0xa8,0x21,0x7e]
; CHECK: fcvtnu    d0, d0            ; encoding: [0x00,0xa8,0x61,0x7e]

  fcvtn   v2.4h, v4.4s
  fcvtn   v3.2s, v5.2d
  fcvtn2  v4.8h, v6.4s
  fcvtn2  v5.4s, v7.2d
  fcvtxn  v6.2s, v9.2d
  fcvtxn2 v7.4s, v8.2d

; CHECK: fcvtn	v2.4h, v4.4s            ; encoding: [0x82,0x68,0x21,0x0e]
; CHECK: fcvtn	v3.2s, v5.2d            ; encoding: [0xa3,0x68,0x61,0x0e]
; CHECK: fcvtn2	v4.8h, v6.4s            ; encoding: [0xc4,0x68,0x21,0x4e]
; CHECK: fcvtn2	v5.4s, v7.2d            ; encoding: [0xe5,0x68,0x61,0x4e]
; CHECK: fcvtxn	v6.2s, v9.2d            ; encoding: [0x26,0x69,0x61,0x2e]
; CHECK: fcvtxn2 v7.4s, v8.2d           ; encoding: [0x07,0x69,0x61,0x6e]

  fcvtps.2s  v0, v0
  fcvtps.4s  v0, v0
  fcvtps.2d  v0, v0
  fcvtps     s0, s0
  fcvtps     d0, d0

; CHECK: fcvtps.2s v0, v0            ; encoding: [0x00,0xa8,0xa1,0x0e]
; CHECK: fcvtps.4s v0, v0            ; encoding: [0x00,0xa8,0xa1,0x4e]
; CHECK: fcvtps.2d v0, v0            ; encoding: [0x00,0xa8,0xe1,0x4e]
; CHECK: fcvtps    s0, s0            ; encoding: [0x00,0xa8,0xa1,0x5e]
; CHECK: fcvtps    d0, d0            ; encoding: [0x00,0xa8,0xe1,0x5e]

  fcvtpu.2s  v0, v0
  fcvtpu.4s  v0, v0
  fcvtpu.2d  v0, v0
  fcvtpu     s0, s0
  fcvtpu     d0, d0

; CHECK: fcvtpu.2s v0, v0            ; encoding: [0x00,0xa8,0xa1,0x2e]
; CHECK: fcvtpu.4s v0, v0            ; encoding: [0x00,0xa8,0xa1,0x6e]
; CHECK: fcvtpu.2d v0, v0            ; encoding: [0x00,0xa8,0xe1,0x6e]
; CHECK: fcvtpu    s0, s0            ; encoding: [0x00,0xa8,0xa1,0x7e]
; CHECK: fcvtpu    d0, d0            ; encoding: [0x00,0xa8,0xe1,0x7e]

  fcvtzs.2s  v0, v0
  fcvtzs.4s  v0, v0
  fcvtzs.2d  v0, v0
  fcvtzs     s0, s0
  fcvtzs     d0, d0

; CHECK: fcvtzs.2s v0, v0            ; encoding: [0x00,0xb8,0xa1,0x0e]
; CHECK: fcvtzs.4s v0, v0            ; encoding: [0x00,0xb8,0xa1,0x4e]
; CHECK: fcvtzs.2d v0, v0            ; encoding: [0x00,0xb8,0xe1,0x4e]
; CHECK: fcvtzs    s0, s0            ; encoding: [0x00,0xb8,0xa1,0x5e]
; CHECK: fcvtzs    d0, d0            ; encoding: [0x00,0xb8,0xe1,0x5e]

  fcvtzu.2s  v0, v0
  fcvtzu.4s  v0, v0
  fcvtzu.2d  v0, v0
  fcvtzu     s0, s0
  fcvtzu     d0, d0

; CHECK: fcvtzu.2s v0, v0            ; encoding: [0x00,0xb8,0xa1,0x2e]
; CHECK: fcvtzu.4s v0, v0            ; encoding: [0x00,0xb8,0xa1,0x6e]
; CHECK: fcvtzu.2d v0, v0            ; encoding: [0x00,0xb8,0xe1,0x6e]
; CHECK: fcvtzu    s0, s0            ; encoding: [0x00,0xb8,0xa1,0x7e]
; CHECK: fcvtzu    d0, d0            ; encoding: [0x00,0xb8,0xe1,0x7e]

;===-------------------------------------------------------------------------===
; AdvSIMD modified immediate instructions
;===-------------------------------------------------------------------------===

  bic.2s  v0, #1
  bic.2s  v0, #1, lsl #0
  bic.2s  v0, #1, lsl #8
  bic.2s  v0, #1, lsl #16
  bic.2s  v0, #1, lsl #24

; CHECK: bic.2s v0, #1               ; encoding: [0x20,0x14,0x00,0x2f]
; CHECK: bic.2s v0, #1               ; encoding: [0x20,0x14,0x00,0x2f]
; CHECK: bic.2s v0, #1, lsl #8       ; encoding: [0x20,0x34,0x00,0x2f]
; CHECK: bic.2s v0, #1, lsl #16      ; encoding: [0x20,0x54,0x00,0x2f]
; CHECK: bic.2s v0, #1, lsl #24      ; encoding: [0x20,0x74,0x00,0x2f]

  bic.4h  v0, #1
  bic.4h  v0, #1, lsl #0
  bic.4h  v0, #1, lsl #8

; CHECK: bic.4h v0, #1               ; encoding: [0x20,0x94,0x00,0x2f]
; CHECK: bic.4h v0, #1               ; encoding: [0x20,0x94,0x00,0x2f]
; CHECK: bic.4h v0, #1, lsl #8       ; encoding: [0x20,0xb4,0x00,0x2f]

  bic.4s  v0, #1
  bic.4s  v0, #1, lsl #0
  bic.4s  v0, #1, lsl #8
  bic.4s  v0, #1, lsl #16
  bic.4s  v0, #1, lsl #24

; CHECK: bic.4s v0, #1               ; encoding: [0x20,0x14,0x00,0x6f]
; CHECK: bic.4s v0, #1               ; encoding: [0x20,0x14,0x00,0x6f]
; CHECK: bic.4s v0, #1, lsl #8       ; encoding: [0x20,0x34,0x00,0x6f]
; CHECK: bic.4s v0, #1, lsl #16      ; encoding: [0x20,0x54,0x00,0x6f]
; CHECK: bic.4s v0, #1, lsl #24      ; encoding: [0x20,0x74,0x00,0x6f]

  bic.8h  v0, #1
  bic.8h  v0, #1, lsl #0
  bic.8h  v0, #1, lsl #8

; CHECK: bic.8h v0, #1               ; encoding: [0x20,0x94,0x00,0x6f]
; CHECK: bic.8h v0, #1               ; encoding: [0x20,0x94,0x00,0x6f]
; CHECK: bic.8h v0, #1, lsl #8       ; encoding: [0x20,0xb4,0x00,0x6f]

  fmov.2d v0, #1.250000e-01

; CHECK: fmov.2d v0, #0.12500000             ; encoding: [0x00,0xf4,0x02,0x6f]

  fmov.2s v0, #1.250000e-01
  fmov.4s v0, #1.250000e-01

; CHECK: fmov.2s v0, #0.12500000             ; encoding: [0x00,0xf4,0x02,0x0f]
; CHECK: fmov.4s v0, #0.12500000             ; encoding: [0x00,0xf4,0x02,0x4f]

  orr.2s  v0, #1
  orr.2s  v0, #1, lsl #0
  orr.2s  v0, #1, lsl #8
  orr.2s  v0, #1, lsl #16
  orr.2s  v0, #1, lsl #24

; CHECK: orr.2s v0, #1               ; encoding: [0x20,0x14,0x00,0x0f]
; CHECK: orr.2s v0, #1               ; encoding: [0x20,0x14,0x00,0x0f]
; CHECK: orr.2s v0, #1, lsl #8       ; encoding: [0x20,0x34,0x00,0x0f]
; CHECK: orr.2s v0, #1, lsl #16      ; encoding: [0x20,0x54,0x00,0x0f]
; CHECK: orr.2s v0, #1, lsl #24      ; encoding: [0x20,0x74,0x00,0x0f]

  orr.4h  v0, #1
  orr.4h  v0, #1, lsl #0
  orr.4h  v0, #1, lsl #8

; CHECK: orr.4h v0, #1               ; encoding: [0x20,0x94,0x00,0x0f]
; CHECK: orr.4h v0, #1               ; encoding: [0x20,0x94,0x00,0x0f]
; CHECK: orr.4h v0, #1, lsl #8       ; encoding: [0x20,0xb4,0x00,0x0f]

  orr.4s  v0, #1
  orr.4s  v0, #1, lsl #0
  orr.4s  v0, #1, lsl #8
  orr.4s  v0, #1, lsl #16
  orr.4s  v0, #1, lsl #24

; CHECK: orr.4s v0, #1               ; encoding: [0x20,0x14,0x00,0x4f]
; CHECK: orr.4s v0, #1               ; encoding: [0x20,0x14,0x00,0x4f]
; CHECK: orr.4s v0, #1, lsl #8       ; encoding: [0x20,0x34,0x00,0x4f]
; CHECK: orr.4s v0, #1, lsl #16      ; encoding: [0x20,0x54,0x00,0x4f]
; CHECK: orr.4s v0, #1, lsl #24      ; encoding: [0x20,0x74,0x00,0x4f]

  orr.8h  v0, #1
  orr.8h  v0, #1, lsl #0
  orr.8h  v0, #1, lsl #8

; CHECK: orr.8h v0, #1               ; encoding: [0x20,0x94,0x00,0x4f]
; CHECK: orr.8h v0, #1               ; encoding: [0x20,0x94,0x00,0x4f]
; CHECK: orr.8h v0, #1, lsl #8       ; encoding: [0x20,0xb4,0x00,0x4f]

  movi     d0, #0x000000000000ff
  movi.2d  v0, #0x000000000000ff

; CHECK: movi     d0, #0x000000000000ff ; encoding: [0x20,0xe4,0x00,0x2f]
; CHECK: movi.2d  v0, #0x000000000000ff ; encoding: [0x20,0xe4,0x00,0x6f]

  movi.2s v0, #1
  movi.2s v0, #1, lsl #0
  movi.2s v0, #1, lsl #8
  movi.2s v0, #1, lsl #16
  movi.2s v0, #1, lsl #24

; CHECK: movi.2s v0, #1              ; encoding: [0x20,0x04,0x00,0x0f]
; CHECK: movi.2s v0, #1              ; encoding: [0x20,0x04,0x00,0x0f]
; CHECK: movi.2s v0, #1, lsl #8      ; encoding: [0x20,0x24,0x00,0x0f]
; CHECK: movi.2s v0, #1, lsl #16     ; encoding: [0x20,0x44,0x00,0x0f]
; CHECK: movi.2s v0, #1, lsl #24     ; encoding: [0x20,0x64,0x00,0x0f]

  movi.4s v0, #1
  movi.4s v0, #1, lsl #0
  movi.4s v0, #1, lsl #8
  movi.4s v0, #1, lsl #16
  movi.4s v0, #1, lsl #24

; CHECK: movi.4s v0, #1              ; encoding: [0x20,0x04,0x00,0x4f]
; CHECK: movi.4s v0, #1              ; encoding: [0x20,0x04,0x00,0x4f]
; CHECK: movi.4s v0, #1, lsl #8      ; encoding: [0x20,0x24,0x00,0x4f]
; CHECK: movi.4s v0, #1, lsl #16     ; encoding: [0x20,0x44,0x00,0x4f]
; CHECK: movi.4s v0, #1, lsl #24     ; encoding: [0x20,0x64,0x00,0x4f]

  movi.4h v0, #1
  movi.4h v0, #1, lsl #0
  movi.4h v0, #1, lsl #8

; CHECK: movi.4h v0, #1              ; encoding: [0x20,0x84,0x00,0x0f]
; CHECK: movi.4h v0, #1              ; encoding: [0x20,0x84,0x00,0x0f]
; CHECK: movi.4h v0, #1, lsl #8      ; encoding: [0x20,0xa4,0x00,0x0f]

  movi.8h v0, #1
  movi.8h v0, #1, lsl #0
  movi.8h v0, #1, lsl #8

; CHECK: movi.8h v0, #1              ; encoding: [0x20,0x84,0x00,0x4f]
; CHECK: movi.8h v0, #1              ; encoding: [0x20,0x84,0x00,0x4f]
; CHECK: movi.8h v0, #1, lsl #8      ; encoding: [0x20,0xa4,0x00,0x4f]

  movi.2s v0, #1, msl #8
  movi.2s v0, #1, msl #16
  movi.4s v0, #1, msl #8
  movi.4s v0, #1, msl #16

; CHECK: movi.2s v0, #1, msl #8      ; encoding: [0x20,0xc4,0x00,0x0f]
; CHECK: movi.2s v0, #1, msl #16     ; encoding: [0x20,0xd4,0x00,0x0f]
; CHECK: movi.4s v0, #1, msl #8      ; encoding: [0x20,0xc4,0x00,0x4f]
; CHECK: movi.4s v0, #1, msl #16     ; encoding: [0x20,0xd4,0x00,0x4f]

  movi.8b  v0, #1
  movi.16b v0, #1

; CHECK: movi.8b  v0, #1             ; encoding: [0x20,0xe4,0x00,0x0f]
; CHECK: movi.16b v0, #1             ; encoding: [0x20,0xe4,0x00,0x4f]

  mvni.2s v0, #1
  mvni.2s v0, #1, lsl #0
  mvni.2s v0, #1, lsl #8
  mvni.2s v0, #1, lsl #16
  mvni.2s v0, #1, lsl #24

; CHECK: mvni.2s v0, #1              ; encoding: [0x20,0x04,0x00,0x2f]
; CHECK: mvni.2s v0, #1              ; encoding: [0x20,0x04,0x00,0x2f]
; CHECK: mvni.2s v0, #1, lsl #8      ; encoding: [0x20,0x24,0x00,0x2f]
; CHECK: mvni.2s v0, #1, lsl #16     ; encoding: [0x20,0x44,0x00,0x2f]
; CHECK: mvni.2s v0, #1, lsl #24     ; encoding: [0x20,0x64,0x00,0x2f]

  mvni.4s v0, #1
  mvni.4s v0, #1, lsl #0
  mvni.4s v0, #1, lsl #8
  mvni.4s v0, #1, lsl #16
  mvni.4s v0, #1, lsl #24

; CHECK: mvni.4s v0, #1              ; encoding: [0x20,0x04,0x00,0x6f]
; CHECK: mvni.4s v0, #1              ; encoding: [0x20,0x04,0x00,0x6f]
; CHECK: mvni.4s v0, #1, lsl #8      ; encoding: [0x20,0x24,0x00,0x6f]
; CHECK: mvni.4s v0, #1, lsl #16     ; encoding: [0x20,0x44,0x00,0x6f]
; CHECK: mvni.4s v0, #1, lsl #24     ; encoding: [0x20,0x64,0x00,0x6f]

  mvni.4h v0, #1
  mvni.4h v0, #1, lsl #0
  mvni.4h v0, #1, lsl #8

; CHECK: mvni.4h v0, #1              ; encoding: [0x20,0x84,0x00,0x2f]
; CHECK: mvni.4h v0, #1              ; encoding: [0x20,0x84,0x00,0x2f]
; CHECK: mvni.4h v0, #1, lsl #8      ; encoding: [0x20,0xa4,0x00,0x2f]

  mvni.8h v0, #1
  mvni.8h v0, #1, lsl #0
  mvni.8h v0, #1, lsl #8

; CHECK: mvni.8h v0, #1              ; encoding: [0x20,0x84,0x00,0x6f]
; CHECK: mvni.8h v0, #1              ; encoding: [0x20,0x84,0x00,0x6f]
; CHECK: mvni.8h v0, #1, lsl #8      ; encoding: [0x20,0xa4,0x00,0x6f]

  mvni.2s v0, #1, msl #8
  mvni.2s v0, #1, msl #16
  mvni.4s v0, #1, msl #8
  mvni.4s v0, #1, msl #16

; CHECK: mvni.2s v0, #1, msl #8      ; encoding: [0x20,0xc4,0x00,0x2f]
; CHECK: mvni.2s v0, #1, msl #16     ; encoding: [0x20,0xd4,0x00,0x2f]
; CHECK: mvni.4s v0, #1, msl #8      ; encoding: [0x20,0xc4,0x00,0x6f]
; CHECK: mvni.4s v0, #1, msl #16     ; encoding: [0x20,0xd4,0x00,0x6f]

;===-------------------------------------------------------------------------===
; AdvSIMD scalar x index
;===-------------------------------------------------------------------------===

  fmla.s  s0, s0, v0[3]
  fmla.d  d0, d0, v0[1]
  fmls.s  s0, s0, v0[3]
  fmls.d  d0, d0, v0[1]
  fmulx.s s0, s0, v0[3]
  fmulx.d d0, d0, v0[1]
  fmul.s  s0, s0, v0[3]
  fmul.d  d0, d0, v0[1]
  sqdmlal.h s0, h0, v0[7]
  sqdmlal.s d0, s0, v0[3]
  sqdmlsl.h s0, h0, v0[7]
  sqdmulh.h h0, h0, v0[7]
  sqdmulh.s s0, s0, v0[3]
  sqdmull.h s0, h0, v0[7]
  sqdmull.s d0, s0, v0[3]
  sqrdmulh.h  h0, h0, v0[7]
  sqrdmulh.s  s0, s0, v0[3]

; CHECK: fmla.s	s0, s0, v0[3]           ; encoding: [0x00,0x18,0xa0,0x5f]
; CHECK: fmla.d	d0, d0, v0[1]           ; encoding: [0x00,0x18,0xc0,0x5f]
; CHECK: fmls.s	s0, s0, v0[3]           ; encoding: [0x00,0x58,0xa0,0x5f]
; CHECK: fmls.d	d0, d0, v0[1]           ; encoding: [0x00,0x58,0xc0,0x5f]
; CHECK: fmulx.s	s0, s0, v0[3]           ; encoding: [0x00,0x98,0xa0,0x7f]
; CHECK: fmulx.d	d0, d0, v0[1]           ; encoding: [0x00,0x98,0xc0,0x7f]
; CHECK: fmul.s	s0, s0, v0[3]           ; encoding: [0x00,0x98,0xa0,0x5f]
; CHECK: fmul.d	d0, d0, v0[1]           ; encoding: [0x00,0x98,0xc0,0x5f]
; CHECK: sqdmlal.h	s0, h0, v0[7]   ; encoding: [0x00,0x38,0x70,0x5f]
; CHECK: sqdmlal.s	d0, s0, v0[3]   ; encoding: [0x00,0x38,0xa0,0x5f]
; CHECK: sqdmlsl.h	s0, h0, v0[7]   ; encoding: [0x00,0x78,0x70,0x5f]
; CHECK: sqdmulh.h	h0, h0, v0[7]   ; encoding: [0x00,0xc8,0x70,0x5f]
; CHECK: sqdmulh.s	s0, s0, v0[3]   ; encoding: [0x00,0xc8,0xa0,0x5f]
; CHECK: sqdmull.h	s0, h0, v0[7]   ; encoding: [0x00,0xb8,0x70,0x5f]
; CHECK: sqdmull.s	d0, s0, v0[3]   ; encoding: [0x00,0xb8,0xa0,0x5f]
; CHECK: sqrdmulh.h	h0, h0, v0[7]   ; encoding: [0x00,0xd8,0x70,0x5f]
; CHECK: sqrdmulh.s	s0, s0, v0[3]   ; encoding: [0x00,0xd8,0xa0,0x5f]

;===-------------------------------------------------------------------------===
; AdvSIMD SMLAL
;===-------------------------------------------------------------------------===
        smlal.8h v1, v2, v3
        smlal.4s v1, v2, v3
        smlal.2d v1, v2, v3
        smlal2.8h v1, v2, v3
        smlal2.4s v1, v2, v3
        smlal2.2d v1, v2, v3

        smlal v13.8h, v8.8b, v0.8b
        smlal v13.4s, v8.4h, v0.4h
        smlal v13.2d, v8.2s, v0.2s
        smlal2 v13.8h, v8.16b, v0.16b
        smlal2 v13.4s, v8.8h, v0.8h
        smlal2 v13.2d, v8.4s, v0.4s

; CHECK: smlal.8h	v1, v2, v3      ; encoding: [0x41,0x80,0x23,0x0e]
; CHECK: smlal.4s	v1, v2, v3      ; encoding: [0x41,0x80,0x63,0x0e]
; CHECK: smlal.2d	v1, v2, v3      ; encoding: [0x41,0x80,0xa3,0x0e]
; CHECK: smlal2.8h	v1, v2, v3      ; encoding: [0x41,0x80,0x23,0x4e]
; CHECK: smlal2.4s	v1, v2, v3      ; encoding: [0x41,0x80,0x63,0x4e]
; CHECK: smlal2.2d	v1, v2, v3      ; encoding: [0x41,0x80,0xa3,0x4e]
; CHECK: smlal.8h	v13, v8, v0     ; encoding: [0x0d,0x81,0x20,0x0e]
; CHECK: smlal.4s	v13, v8, v0     ; encoding: [0x0d,0x81,0x60,0x0e]
; CHECK: smlal.2d	v13, v8, v0     ; encoding: [0x0d,0x81,0xa0,0x0e]
; CHECK: smlal2.8h	v13, v8, v0     ; encoding: [0x0d,0x81,0x20,0x4e]
; CHECK: smlal2.4s	v13, v8, v0     ; encoding: [0x0d,0x81,0x60,0x4e]
; CHECK: smlal2.2d	v13, v8, v0     ; encoding: [0x0d,0x81,0xa0,0x4e]


;===-------------------------------------------------------------------------===
; AdvSIMD scalar x index
;===-------------------------------------------------------------------------===

  fmla.2s v0, v0, v0[0]
  fmla.4s v0, v0, v0[1]
  fmla.2d v0, v0, v0[1]
  fmls.2s v0, v0, v0[0]
  fmls.4s v0, v0, v0[1]
  fmls.2d v0, v0, v0[1]
  fmulx.2s  v0, v0, v0[0]
  fmulx.4s  v0, v0, v0[1]
  fmulx.2d  v0, v0, v0[1]
  fmul.2s v0, v0, v0[0]
  fmul.4s v0, v0, v0[1]
  fmul.2d v0, v0, v0[1]
  mla.4h  v0, v0, v0[0]
  mla.8h  v0, v0, v0[1]
  mla.2s  v0, v0, v0[2]
  mla.4s  v0, v0, v0[3]
  mls.4h  v0, v0, v0[0]
  mls.8h  v0, v0, v0[1]
  mls.2s  v0, v0, v0[2]
  mls.4s  v0, v0, v0[3]
  mul.4h  v0, v0, v0[0]
  mul.8h  v0, v0, v0[1]
  mul.2s  v0, v0, v0[2]
  mul.4s  v0, v0, v0[3]
  smlal.4s  v0, v0, v0[0]
  smlal2.4s v0, v0, v0[1]
  smlal.2d  v0, v0, v0[2]
  smlal2.2d v0, v0, v0[3]
  smlsl.4s  v0, v0, v0[0]
  smlsl2.4s v0, v0, v0[1]
  smlsl.2d  v0, v0, v0[2]
  smlsl2.2d v0, v0, v0[3]
  smull.4s  v0, v0, v0[0]
  smull2.4s v0, v0, v0[1]
  smull.2d  v0, v0, v0[2]
  smull2.2d v0, v0, v0[3]
  sqdmlal.4s  v0, v0, v0[0]
  sqdmlal2.4s v0, v0, v0[1]
  sqdmlal.2d  v0, v0, v0[2]
  sqdmlal2.2d v0, v0, v0[3]
  sqdmlsl.4s  v0, v0, v0[0]
  sqdmlsl2.4s v0, v0, v0[1]
  sqdmlsl.2d  v0, v0, v0[2]
  sqdmlsl2.2d v0, v0, v0[3]
  sqdmulh.4h  v0, v0, v0[0]
  sqdmulh.8h  v0, v0, v0[1]
  sqdmulh.2s  v0, v0, v0[2]
  sqdmulh.4s  v0, v0, v0[3]
  sqdmull.4s  v0, v0, v0[0]
  sqdmull2.4s v0, v0, v0[1]
  sqdmull.2d  v0, v0, v0[2]
  sqdmull2.2d v0, v0, v0[3]
  sqrdmulh.4h v0, v0, v0[0]
  sqrdmulh.8h v0, v0, v0[1]
  sqrdmulh.2s v0, v0, v0[2]
  sqrdmulh.4s v0, v0, v0[3]
  umlal.4s  v0, v0, v0[0]
  umlal2.4s v0, v0, v0[1]
  umlal.2d  v0, v0, v0[2]
  umlal2.2d v0, v0, v0[3]
  umlsl.4s  v0, v0, v0[0]
  umlsl2.4s v0, v0, v0[1]
  umlsl.2d  v0, v0, v0[2]
  umlsl2.2d v0, v0, v0[3]
  umull.4s  v0, v0, v0[0]
  umull2.4s v0, v0, v0[1]
  umull.2d  v0, v0, v0[2]
  umull2.2d v0, v0, v0[3]

; CHECK: fmla.2s	v0, v0, v0[0]           ; encoding: [0x00,0x10,0x80,0x0f]
; CHECK: fmla.4s	v0, v0, v0[1]           ; encoding: [0x00,0x10,0xa0,0x4f]
; CHECK: fmla.2d	v0, v0, v0[1]           ; encoding: [0x00,0x18,0xc0,0x4f]
; CHECK: fmls.2s	v0, v0, v0[0]           ; encoding: [0x00,0x50,0x80,0x0f]
; CHECK: fmls.4s	v0, v0, v0[1]           ; encoding: [0x00,0x50,0xa0,0x4f]
; CHECK: fmls.2d	v0, v0, v0[1]           ; encoding: [0x00,0x58,0xc0,0x4f]
; CHECK: fmulx.2s	v0, v0, v0[0]   ; encoding: [0x00,0x90,0x80,0x2f]
; CHECK: fmulx.4s	v0, v0, v0[1]   ; encoding: [0x00,0x90,0xa0,0x6f]
; CHECK: fmulx.2d	v0, v0, v0[1]   ; encoding: [0x00,0x98,0xc0,0x6f]
; CHECK: fmul.2s	v0, v0, v0[0]           ; encoding: [0x00,0x90,0x80,0x0f]
; CHECK: fmul.4s	v0, v0, v0[1]           ; encoding: [0x00,0x90,0xa0,0x4f]
; CHECK: fmul.2d	v0, v0, v0[1]           ; encoding: [0x00,0x98,0xc0,0x4f]
; CHECK: mla.4h	v0, v0, v0[0]           ; encoding: [0x00,0x00,0x40,0x2f]
; CHECK: mla.8h	v0, v0, v0[1]           ; encoding: [0x00,0x00,0x50,0x6f]
; CHECK: mla.2s	v0, v0, v0[2]           ; encoding: [0x00,0x08,0x80,0x2f]
; CHECK: mla.4s	v0, v0, v0[3]           ; encoding: [0x00,0x08,0xa0,0x6f]
; CHECK: mls.4h	v0, v0, v0[0]           ; encoding: [0x00,0x40,0x40,0x2f]
; CHECK: mls.8h	v0, v0, v0[1]           ; encoding: [0x00,0x40,0x50,0x6f]
; CHECK: mls.2s	v0, v0, v0[2]           ; encoding: [0x00,0x48,0x80,0x2f]
; CHECK: mls.4s	v0, v0, v0[3]           ; encoding: [0x00,0x48,0xa0,0x6f]
; CHECK: mul.4h	v0, v0, v0[0]           ; encoding: [0x00,0x80,0x40,0x0f]
; CHECK: mul.8h	v0, v0, v0[1]           ; encoding: [0x00,0x80,0x50,0x4f]
; CHECK: mul.2s	v0, v0, v0[2]           ; encoding: [0x00,0x88,0x80,0x0f]
; CHECK: mul.4s	v0, v0, v0[3]           ; encoding: [0x00,0x88,0xa0,0x4f]
; CHECK: smlal.4s	v0, v0, v0[0]   ; encoding: [0x00,0x20,0x40,0x0f]
; CHECK: smlal2.4s	v0, v0, v0[1]   ; encoding: [0x00,0x20,0x50,0x4f]
; CHECK: smlal.2d	v0, v0, v0[2]   ; encoding: [0x00,0x28,0x80,0x0f]
; CHECK: smlal2.2d	v0, v0, v0[3]   ; encoding: [0x00,0x28,0xa0,0x4f]
; CHECK: smlsl.4s	v0, v0, v0[0]   ; encoding: [0x00,0x60,0x40,0x0f]
; CHECK: smlsl2.4s	v0, v0, v0[1]   ; encoding: [0x00,0x60,0x50,0x4f]
; CHECK: smlsl.2d	v0, v0, v0[2]   ; encoding: [0x00,0x68,0x80,0x0f]
; CHECK: smlsl2.2d	v0, v0, v0[3]   ; encoding: [0x00,0x68,0xa0,0x4f]
; CHECK: smull.4s	v0, v0, v0[0]   ; encoding: [0x00,0xa0,0x40,0x0f]
; CHECK: smull2.4s	v0, v0, v0[1]   ; encoding: [0x00,0xa0,0x50,0x4f]
; CHECK: smull.2d	v0, v0, v0[2]   ; encoding: [0x00,0xa8,0x80,0x0f]
; CHECK: smull2.2d	v0, v0, v0[3]   ; encoding: [0x00,0xa8,0xa0,0x4f]
; CHECK: sqdmlal.4s	v0, v0, v0[0]   ; encoding: [0x00,0x30,0x40,0x0f]
; CHECK: sqdmlal2.4s	v0, v0, v0[1]   ; encoding: [0x00,0x30,0x50,0x4f]
; CHECK: sqdmlal.2d	v0, v0, v0[2]   ; encoding: [0x00,0x38,0x80,0x0f]
; CHECK: sqdmlal2.2d	v0, v0, v0[3]   ; encoding: [0x00,0x38,0xa0,0x4f]
; CHECK: sqdmlsl.4s	v0, v0, v0[0]   ; encoding: [0x00,0x70,0x40,0x0f]
; CHECK: sqdmlsl2.4s	v0, v0, v0[1]   ; encoding: [0x00,0x70,0x50,0x4f]
; CHECK: sqdmlsl.2d	v0, v0, v0[2]   ; encoding: [0x00,0x78,0x80,0x0f]
; CHECK: sqdmlsl2.2d	v0, v0, v0[3]   ; encoding: [0x00,0x78,0xa0,0x4f]
; CHECK: sqdmulh.4h	v0, v0, v0[0]   ; encoding: [0x00,0xc0,0x40,0x0f]
; CHECK: sqdmulh.8h	v0, v0, v0[1]   ; encoding: [0x00,0xc0,0x50,0x4f]
; CHECK: sqdmulh.2s	v0, v0, v0[2]   ; encoding: [0x00,0xc8,0x80,0x0f]
; CHECK: sqdmulh.4s	v0, v0, v0[3]   ; encoding: [0x00,0xc8,0xa0,0x4f]
; CHECK: sqdmull.4s	v0, v0, v0[0]   ; encoding: [0x00,0xb0,0x40,0x0f]
; CHECK: sqdmull2.4s	v0, v0, v0[1]   ; encoding: [0x00,0xb0,0x50,0x4f]
; CHECK: sqdmull.2d	v0, v0, v0[2]   ; encoding: [0x00,0xb8,0x80,0x0f]
; CHECK: sqdmull2.2d	v0, v0, v0[3]   ; encoding: [0x00,0xb8,0xa0,0x4f]
; CHECK: sqrdmulh.4h	v0, v0, v0[0]   ; encoding: [0x00,0xd0,0x40,0x0f]
; CHECK: sqrdmulh.8h	v0, v0, v0[1]   ; encoding: [0x00,0xd0,0x50,0x4f]
; CHECK: sqrdmulh.2s	v0, v0, v0[2]   ; encoding: [0x00,0xd8,0x80,0x0f]
; CHECK: sqrdmulh.4s	v0, v0, v0[3]   ; encoding: [0x00,0xd8,0xa0,0x4f]
; CHECK: umlal.4s	v0, v0, v0[0]   ; encoding: [0x00,0x20,0x40,0x2f]
; CHECK: umlal2.4s	v0, v0, v0[1]   ; encoding: [0x00,0x20,0x50,0x6f]
; CHECK: umlal.2d	v0, v0, v0[2]   ; encoding: [0x00,0x28,0x80,0x2f]
; CHECK: umlal2.2d	v0, v0, v0[3]   ; encoding: [0x00,0x28,0xa0,0x6f]
; CHECK: umlsl.4s	v0, v0, v0[0]   ; encoding: [0x00,0x60,0x40,0x2f]
; CHECK: umlsl2.4s	v0, v0, v0[1]   ; encoding: [0x00,0x60,0x50,0x6f]
; CHECK: umlsl.2d	v0, v0, v0[2]   ; encoding: [0x00,0x68,0x80,0x2f]
; CHECK: umlsl2.2d	v0, v0, v0[3]   ; encoding: [0x00,0x68,0xa0,0x6f]
; CHECK: umull.4s	v0, v0, v0[0]   ; encoding: [0x00,0xa0,0x40,0x2f]
; CHECK: umull2.4s	v0, v0, v0[1]   ; encoding: [0x00,0xa0,0x50,0x6f]
; CHECK: umull.2d	v0, v0, v0[2]   ; encoding: [0x00,0xa8,0x80,0x2f]
; CHECK: umull2.2d	v0, v0, v0[3]   ; encoding: [0x00,0xa8,0xa0,0x6f]


;===-------------------------------------------------------------------------===
; AdvSIMD scalar with shift
;===-------------------------------------------------------------------------===

  fcvtzs s0, s0, #1
  fcvtzs d0, d0, #2
  fcvtzu s0, s0, #1
  fcvtzu d0, d0, #2
  shl    d0, d0, #1
  sli    d0, d0, #1
  sqrshrn b0, h0, #1
  sqrshrn h0, s0, #2
  sqrshrn s0, d0, #3
  sqrshrun b0, h0, #1
  sqrshrun h0, s0, #2
  sqrshrun s0, d0, #3
  sqshlu  b0, b0, #1
  sqshlu  h0, h0, #2
  sqshlu  s0, s0, #3
  sqshlu  d0, d0, #4
  sqshl   b0, b0, #1
  sqshl   h0, h0, #2
  sqshl   s0, s0, #3
  sqshl   d0, d0, #4
  sqshrn  b0, h0, #1
  sqshrn  h0, s0, #2
  sqshrn  s0, d0, #3
  sqshrun b0, h0, #1
  sqshrun h0, s0, #2
  sqshrun s0, d0, #3
  sri     d0, d0, #1
  srshr   d0, d0, #1
  srsra   d0, d0, #1
  sshr    d0, d0, #1
  ucvtf   s0, s0, #1
  ucvtf   d0, d0, #2
  scvtf   s0, s0, #1
  scvtf   d0, d0, #2
  uqrshrn b0, h0, #1
  uqrshrn h0, s0, #2
  uqrshrn s0, d0, #3
  uqshl   b0, b0, #1
  uqshl   h0, h0, #2
  uqshl   s0, s0, #3
  uqshl   d0, d0, #4
  uqshrn  b0, h0, #1
  uqshrn  h0, s0, #2
  uqshrn  s0, d0, #3
  urshr   d0, d0, #1
  ursra   d0, d0, #1
  ushr    d0, d0, #1
  usra    d0, d0, #1

; CHECK: fcvtzs	s0, s0, #1              ; encoding: [0x00,0xfc,0x3f,0x5f]
; CHECK: fcvtzs	d0, d0, #2              ; encoding: [0x00,0xfc,0x7e,0x5f]
; CHECK: fcvtzu	s0, s0, #1              ; encoding: [0x00,0xfc,0x3f,0x7f]
; CHECK: fcvtzu	d0, d0, #2              ; encoding: [0x00,0xfc,0x7e,0x7f]
; CHECK: shl	d0, d0, #1              ; encoding: [0x00,0x54,0x41,0x5f]
; CHECK: sli	d0, d0, #1              ; encoding: [0x00,0x54,0x41,0x7f]
; CHECK: sqrshrn	b0, h0, #1              ; encoding: [0x00,0x9c,0x0f,0x5f]
; CHECK: sqrshrn	h0, s0, #2              ; encoding: [0x00,0x9c,0x1e,0x5f]
; CHECK: sqrshrn	s0, d0, #3              ; encoding: [0x00,0x9c,0x3d,0x5f]
; CHECK: sqrshrun	b0, h0, #1      ; encoding: [0x00,0x8c,0x0f,0x7f]
; CHECK: sqrshrun	h0, s0, #2      ; encoding: [0x00,0x8c,0x1e,0x7f]
; CHECK: sqrshrun	s0, d0, #3      ; encoding: [0x00,0x8c,0x3d,0x7f]
; CHECK: sqshlu	b0, b0, #1              ; encoding: [0x00,0x64,0x09,0x7f]
; CHECK: sqshlu	h0, h0, #2              ; encoding: [0x00,0x64,0x12,0x7f]
; CHECK: sqshlu	s0, s0, #3              ; encoding: [0x00,0x64,0x23,0x7f]
; CHECK: sqshlu	d0, d0, #4              ; encoding: [0x00,0x64,0x44,0x7f]
; CHECK: sqshl	b0, b0, #1              ; encoding: [0x00,0x74,0x09,0x5f]
; CHECK: sqshl	h0, h0, #2              ; encoding: [0x00,0x74,0x12,0x5f]
; CHECK: sqshl	s0, s0, #3              ; encoding: [0x00,0x74,0x23,0x5f]
; CHECK: sqshl	d0, d0, #4              ; encoding: [0x00,0x74,0x44,0x5f]
; CHECK: sqshrn	b0, h0, #1              ; encoding: [0x00,0x94,0x0f,0x5f]
; CHECK: sqshrn	h0, s0, #2              ; encoding: [0x00,0x94,0x1e,0x5f]
; CHECK: sqshrn	s0, d0, #3              ; encoding: [0x00,0x94,0x3d,0x5f]
; CHECK: sqshrun	b0, h0, #1              ; encoding: [0x00,0x84,0x0f,0x7f]
; CHECK: sqshrun	h0, s0, #2              ; encoding: [0x00,0x84,0x1e,0x7f]
; CHECK: sqshrun	s0, d0, #3              ; encoding: [0x00,0x84,0x3d,0x7f]
; CHECK: sri	d0, d0, #1              ; encoding: [0x00,0x44,0x7f,0x7f]
; CHECK: srshr	d0, d0, #1              ; encoding: [0x00,0x24,0x7f,0x5f]
; CHECK: srsra	d0, d0, #1              ; encoding: [0x00,0x34,0x7f,0x5f]
; CHECK: sshr	d0, d0, #1              ; encoding: [0x00,0x04,0x7f,0x5f]
; CHECK: ucvtf	s0, s0, #1              ; encoding: [0x00,0xe4,0x3f,0x7f]
; CHECK: ucvtf	d0, d0, #2              ; encoding: [0x00,0xe4,0x7e,0x7f]
; check: scvtf  s0, s0, #1              ; encoding: [0x00,0xe4,0x3f,0x5f]
; check: scvtf  d0, d0, #2              ; encoding: [0x00,0xe4,0x7e,0x5f]
; CHECK: uqrshrn	b0, h0, #1              ; encoding: [0x00,0x9c,0x0f,0x7f]
; CHECK: uqrshrn	h0, s0, #2              ; encoding: [0x00,0x9c,0x1e,0x7f]
; CHECK: uqrshrn	s0, d0, #3              ; encoding: [0x00,0x9c,0x3d,0x7f]
; CHECK: uqshl	b0, b0, #1              ; encoding: [0x00,0x74,0x09,0x7f]
; CHECK: uqshl	h0, h0, #2              ; encoding: [0x00,0x74,0x12,0x7f]
; CHECK: uqshl	s0, s0, #3              ; encoding: [0x00,0x74,0x23,0x7f]
; CHECK: uqshl	d0, d0, #4              ; encoding: [0x00,0x74,0x44,0x7f]
; CHECK: uqshrn	b0, h0, #1              ; encoding: [0x00,0x94,0x0f,0x7f]
; CHECK: uqshrn	h0, s0, #2              ; encoding: [0x00,0x94,0x1e,0x7f]
; CHECK: uqshrn	s0, d0, #3              ; encoding: [0x00,0x94,0x3d,0x7f]
; CHECK: urshr	d0, d0, #1              ; encoding: [0x00,0x24,0x7f,0x7f]
; CHECK: ursra	d0, d0, #1              ; encoding: [0x00,0x34,0x7f,0x7f]
; CHECK: ushr	d0, d0, #1              ; encoding: [0x00,0x04,0x7f,0x7f]
; CHECK: usra	d0, d0, #1              ; encoding: [0x00,0x14,0x7f,0x7f]


;===-------------------------------------------------------------------------===
; AdvSIMD vector with shift
;===-------------------------------------------------------------------------===

   fcvtzs.2s v0, v0, #1
   fcvtzs.4s v0, v0, #2
   fcvtzs.2d v0, v0, #3
   fcvtzu.2s v0, v0, #1
   fcvtzu.4s v0, v0, #2
   fcvtzu.2d v0, v0, #3
   rshrn.8b v0, v0, #1
   rshrn2.16b v0, v0, #2
   rshrn.4h v0, v0, #3
   rshrn2.8h v0, v0, #4
   rshrn.2s v0, v0, #5
   rshrn2.4s v0, v0, #6
   scvtf.2s v0, v0, #1
   scvtf.4s v0, v0, #2
   scvtf.2d v0, v0, #3
   shl.8b v0, v0, #1
   shl.16b v0, v0, #2
   shl.4h v0, v0, #3
   shl.8h v0, v0, #4
   shl.2s v0, v0, #5
   shl.4s v0, v0, #6
   shl.2d v0, v0, #7
   shrn.8b v0, v0, #1
   shrn2.16b v0, v0, #2
   shrn.4h v0, v0, #3
   shrn2.8h v0, v0, #4
   shrn.2s v0, v0, #5
   shrn2.4s v0, v0, #6
   sli.8b v0, v0, #1
   sli.16b v0, v0, #2
   sli.4h v0, v0, #3
   sli.8h v0, v0, #4
   sli.2s v0, v0, #5
   sli.4s v0, v0, #6
   sli.2d v0, v0, #7
   sqrshrn.8b v0, v0, #1
   sqrshrn2.16b v0, v0, #2
   sqrshrn.4h v0, v0, #3
   sqrshrn2.8h v0, v0, #4
   sqrshrn.2s v0, v0, #5
   sqrshrn2.4s v0, v0, #6
   sqrshrun.8b v0, v0, #1
   sqrshrun2.16b v0, v0, #2
   sqrshrun.4h v0, v0, #3
   sqrshrun2.8h v0, v0, #4
   sqrshrun.2s v0, v0, #5
   sqrshrun2.4s v0, v0, #6
   sqshlu.8b v0, v0, #1
   sqshlu.16b v0, v0, #2
   sqshlu.4h v0, v0, #3
   sqshlu.8h v0, v0, #4
   sqshlu.2s v0, v0, #5
   sqshlu.4s v0, v0, #6
   sqshlu.2d v0, v0, #7
   sqshl.8b v0, v0, #1
   sqshl.16b v0, v0, #2
   sqshl.4h v0, v0, #3
   sqshl.8h v0, v0, #4
   sqshl.2s v0, v0, #5
   sqshl.4s v0, v0, #6
   sqshl.2d v0, v0, #7
   sqshrn.8b v0, v0, #1
   sqshrn2.16b v0, v0, #2
   sqshrn.4h v0, v0, #3
   sqshrn2.8h v0, v0, #4
   sqshrn.2s v0, v0, #5
   sqshrn2.4s v0, v0, #6
   sqshrun.8b v0, v0, #1
   sqshrun2.16b v0, v0, #2
   sqshrun.4h v0, v0, #3
   sqshrun2.8h v0, v0, #4
   sqshrun.2s v0, v0, #5
   sqshrun2.4s v0, v0, #6
   sri.8b v0, v0, #1
   sri.16b v0, v0, #2
   sri.4h v0, v0, #3
   sri.8h v0, v0, #4
   sri.2s v0, v0, #5
   sri.4s v0, v0, #6
   sri.2d v0, v0, #7
   srshr.8b v0, v0, #1
   srshr.16b v0, v0, #2
   srshr.4h v0, v0, #3
   srshr.8h v0, v0, #4
   srshr.2s v0, v0, #5
   srshr.4s v0, v0, #6
   srshr.2d v0, v0, #7
   srsra.8b v0, v0, #1
   srsra.16b v0, v0, #2
   srsra.4h v0, v0, #3
   srsra.8h v0, v0, #4
   srsra.2s v0, v0, #5
   srsra.4s v0, v0, #6
   srsra.2d v0, v0, #7
   sshll.8h v0, v0, #1
   sshll2.8h v0, v0, #2
   sshll.4s v0, v0, #3
   sshll2.4s v0, v0, #4
   sshll.2d v0, v0, #5
   sshll2.2d v0, v0, #6
   sshr.8b v0, v0, #1
   sshr.16b v0, v0, #2
   sshr.4h v0, v0, #3
   sshr.8h v0, v0, #4
   sshr.2s v0, v0, #5
   sshr.4s v0, v0, #6
   sshr.2d v0, v0, #7
   sshr.8b v0, v0, #1
   ssra.16b v0, v0, #2
   ssra.4h v0, v0, #3
   ssra.8h v0, v0, #4
   ssra.2s v0, v0, #5
   ssra.4s v0, v0, #6
   ssra.2d v0, v0, #7
   ssra d0, d0, #64
   ucvtf.2s v0, v0, #1
   ucvtf.4s v0, v0, #2
   ucvtf.2d v0, v0, #3
   uqrshrn.8b v0, v0, #1
   uqrshrn2.16b v0, v0, #2
   uqrshrn.4h v0, v0, #3
   uqrshrn2.8h v0, v0, #4
   uqrshrn.2s v0, v0, #5
   uqrshrn2.4s v0, v0, #6
   uqshl.8b v0, v0, #1
   uqshl.16b v0, v0, #2
   uqshl.4h v0, v0, #3
   uqshl.8h v0, v0, #4
   uqshl.2s v0, v0, #5
   uqshl.4s v0, v0, #6
   uqshl.2d v0, v0, #7
   uqshrn.8b v0, v0, #1
   uqshrn2.16b v0, v0, #2
   uqshrn.4h v0, v0, #3
   uqshrn2.8h v0, v0, #4
   uqshrn.2s v0, v0, #5
   uqshrn2.4s v0, v0, #6
   urshr.8b v0, v0, #1
   urshr.16b v0, v0, #2
   urshr.4h v0, v0, #3
   urshr.8h v0, v0, #4
   urshr.2s v0, v0, #5
   urshr.4s v0, v0, #6
   urshr.2d v0, v0, #7
   ursra.8b v0, v0, #1
   ursra.16b v0, v0, #2
   ursra.4h v0, v0, #3
   ursra.8h v0, v0, #4
   ursra.2s v0, v0, #5
   ursra.4s v0, v0, #6
   ursra.2d v0, v0, #7
   ushll.8h v0, v0, #1
   ushll2.8h v0, v0, #2
   ushll.4s v0, v0, #3
   ushll2.4s v0, v0, #4
   ushll.2d v0, v0, #5
   ushll2.2d v0, v0, #6
   ushr.8b v0, v0, #1
   ushr.16b v0, v0, #2
   ushr.4h v0, v0, #3
   ushr.8h v0, v0, #4
   ushr.2s v0, v0, #5
   ushr.4s v0, v0, #6
   ushr.2d v0, v0, #7
   usra.8b v0, v0, #1
   usra.16b v0, v0, #2
   usra.4h v0, v0, #3
   usra.8h v0, v0, #4
   usra.2s v0, v0, #5
   usra.4s v0, v0, #6
   usra.2d v0, v0, #7

; CHECK: fcvtzs.2s	v0, v0, #1      ; encoding: [0x00,0xfc,0x3f,0x0f]
; CHECK: fcvtzs.4s	v0, v0, #2      ; encoding: [0x00,0xfc,0x3e,0x4f]
; CHECK: fcvtzs.2d	v0, v0, #3      ; encoding: [0x00,0xfc,0x7d,0x4f]
; CHECK: fcvtzu.2s	v0, v0, #1      ; encoding: [0x00,0xfc,0x3f,0x2f]
; CHECK: fcvtzu.4s	v0, v0, #2      ; encoding: [0x00,0xfc,0x3e,0x6f]
; CHECK: fcvtzu.2d	v0, v0, #3      ; encoding: [0x00,0xfc,0x7d,0x6f]
; CHECK: rshrn.8b	v0, v0, #1      ; encoding: [0x00,0x8c,0x0f,0x0f]
; CHECK: rshrn2.16b	v0, v0, #2      ; encoding: [0x00,0x8c,0x0e,0x4f]
; CHECK: rshrn.4h	v0, v0, #3      ; encoding: [0x00,0x8c,0x1d,0x0f]
; CHECK: rshrn2.8h	v0, v0, #4      ; encoding: [0x00,0x8c,0x1c,0x4f]
; CHECK: rshrn.2s	v0, v0, #5      ; encoding: [0x00,0x8c,0x3b,0x0f]
; CHECK: rshrn2.4s	v0, v0, #6      ; encoding: [0x00,0x8c,0x3a,0x4f]
; CHECK: scvtf.2s	v0, v0, #1      ; encoding: [0x00,0xe4,0x3f,0x0f]
; CHECK: scvtf.4s	v0, v0, #2      ; encoding: [0x00,0xe4,0x3e,0x4f]
; CHECK: scvtf.2d	v0, v0, #3      ; encoding: [0x00,0xe4,0x7d,0x4f]
; CHECK: shl.8b	v0, v0, #1              ; encoding: [0x00,0x54,0x09,0x0f]
; CHECK: shl.16b	v0, v0, #2              ; encoding: [0x00,0x54,0x0a,0x4f]
; CHECK: shl.4h	v0, v0, #3              ; encoding: [0x00,0x54,0x13,0x0f]
; CHECK: shl.8h	v0, v0, #4              ; encoding: [0x00,0x54,0x14,0x4f]
; CHECK: shl.2s	v0, v0, #5              ; encoding: [0x00,0x54,0x25,0x0f]
; CHECK: shl.4s	v0, v0, #6              ; encoding: [0x00,0x54,0x26,0x4f]
; CHECK: shl.2d	v0, v0, #7              ; encoding: [0x00,0x54,0x47,0x4f]
; CHECK: shrn.8b	v0, v0, #1              ; encoding: [0x00,0x84,0x0f,0x0f]
; CHECK: shrn2.16b	v0, v0, #2      ; encoding: [0x00,0x84,0x0e,0x4f]
; CHECK: shrn.4h	v0, v0, #3              ; encoding: [0x00,0x84,0x1d,0x0f]
; CHECK: shrn2.8h	v0, v0, #4      ; encoding: [0x00,0x84,0x1c,0x4f]
; CHECK: shrn.2s	v0, v0, #5              ; encoding: [0x00,0x84,0x3b,0x0f]
; CHECK: shrn2.4s	v0, v0, #6      ; encoding: [0x00,0x84,0x3a,0x4f]
; CHECK: sli.8b	v0, v0, #1              ; encoding: [0x00,0x54,0x09,0x2f]
; CHECK: sli.16b	v0, v0, #2              ; encoding: [0x00,0x54,0x0a,0x6f]
; CHECK: sli.4h	v0, v0, #3              ; encoding: [0x00,0x54,0x13,0x2f]
; CHECK: sli.8h	v0, v0, #4              ; encoding: [0x00,0x54,0x14,0x6f]
; CHECK: sli.2s	v0, v0, #5              ; encoding: [0x00,0x54,0x25,0x2f]
; CHECK: sli.4s	v0, v0, #6              ; encoding: [0x00,0x54,0x26,0x6f]
; CHECK: sli.2d	v0, v0, #7              ; encoding: [0x00,0x54,0x47,0x6f]
; CHECK: sqrshrn.8b	v0, v0, #1      ; encoding: [0x00,0x9c,0x0f,0x0f]
; CHECK: sqrshrn2.16b	v0, v0, #2      ; encoding: [0x00,0x9c,0x0e,0x4f]
; CHECK: sqrshrn.4h	v0, v0, #3      ; encoding: [0x00,0x9c,0x1d,0x0f]
; CHECK: sqrshrn2.8h	v0, v0, #4      ; encoding: [0x00,0x9c,0x1c,0x4f]
; CHECK: sqrshrn.2s	v0, v0, #5      ; encoding: [0x00,0x9c,0x3b,0x0f]
; CHECK: sqrshrn2.4s	v0, v0, #6      ; encoding: [0x00,0x9c,0x3a,0x4f]
; CHECK: sqrshrun.8b	v0, v0, #1      ; encoding: [0x00,0x8c,0x0f,0x2f]
; CHECK: sqrshrun2.16b	v0, v0, #2      ; encoding: [0x00,0x8c,0x0e,0x6f]
; CHECK: sqrshrun.4h	v0, v0, #3      ; encoding: [0x00,0x8c,0x1d,0x2f]
; CHECK: sqrshrun2.8h	v0, v0, #4      ; encoding: [0x00,0x8c,0x1c,0x6f]
; CHECK: sqrshrun.2s	v0, v0, #5      ; encoding: [0x00,0x8c,0x3b,0x2f]
; CHECK: sqrshrun2.4s	v0, v0, #6      ; encoding: [0x00,0x8c,0x3a,0x6f]
; CHECK: sqshlu.8b	v0, v0, #1      ; encoding: [0x00,0x64,0x09,0x2f]
; CHECK: sqshlu.16b	v0, v0, #2      ; encoding: [0x00,0x64,0x0a,0x6f]
; CHECK: sqshlu.4h	v0, v0, #3      ; encoding: [0x00,0x64,0x13,0x2f]
; CHECK: sqshlu.8h	v0, v0, #4      ; encoding: [0x00,0x64,0x14,0x6f]
; CHECK: sqshlu.2s	v0, v0, #5      ; encoding: [0x00,0x64,0x25,0x2f]
; CHECK: sqshlu.4s	v0, v0, #6      ; encoding: [0x00,0x64,0x26,0x6f]
; CHECK: sqshlu.2d	v0, v0, #7      ; encoding: [0x00,0x64,0x47,0x6f]
; CHECK: sqshl.8b	v0, v0, #1      ; encoding: [0x00,0x74,0x09,0x0f]
; CHECK: sqshl.16b	v0, v0, #2      ; encoding: [0x00,0x74,0x0a,0x4f]
; CHECK: sqshl.4h	v0, v0, #3      ; encoding: [0x00,0x74,0x13,0x0f]
; CHECK: sqshl.8h	v0, v0, #4      ; encoding: [0x00,0x74,0x14,0x4f]
; CHECK: sqshl.2s	v0, v0, #5      ; encoding: [0x00,0x74,0x25,0x0f]
; CHECK: sqshl.4s	v0, v0, #6      ; encoding: [0x00,0x74,0x26,0x4f]
; CHECK: sqshl.2d	v0, v0, #7      ; encoding: [0x00,0x74,0x47,0x4f]
; CHECK: sqshrn.8b	v0, v0, #1      ; encoding: [0x00,0x94,0x0f,0x0f]
; CHECK: sqshrn2.16b	v0, v0, #2      ; encoding: [0x00,0x94,0x0e,0x4f]
; CHECK: sqshrn.4h	v0, v0, #3      ; encoding: [0x00,0x94,0x1d,0x0f]
; CHECK: sqshrn2.8h	v0, v0, #4      ; encoding: [0x00,0x94,0x1c,0x4f]
; CHECK: sqshrn.2s	v0, v0, #5      ; encoding: [0x00,0x94,0x3b,0x0f]
; CHECK: sqshrn2.4s	v0, v0, #6      ; encoding: [0x00,0x94,0x3a,0x4f]
; CHECK: sqshrun.8b	v0, v0, #1      ; encoding: [0x00,0x84,0x0f,0x2f]
; CHECK: sqshrun2.16b	v0, v0, #2      ; encoding: [0x00,0x84,0x0e,0x6f]
; CHECK: sqshrun.4h	v0, v0, #3      ; encoding: [0x00,0x84,0x1d,0x2f]
; CHECK: sqshrun2.8h	v0, v0, #4      ; encoding: [0x00,0x84,0x1c,0x6f]
; CHECK: sqshrun.2s	v0, v0, #5      ; encoding: [0x00,0x84,0x3b,0x2f]
; CHECK: sqshrun2.4s	v0, v0, #6      ; encoding: [0x00,0x84,0x3a,0x6f]
; CHECK: sri.8b	v0, v0, #1              ; encoding: [0x00,0x44,0x0f,0x2f]
; CHECK: sri.16b	v0, v0, #2              ; encoding: [0x00,0x44,0x0e,0x6f]
; CHECK: sri.4h	v0, v0, #3              ; encoding: [0x00,0x44,0x1d,0x2f]
; CHECK: sri.8h	v0, v0, #4              ; encoding: [0x00,0x44,0x1c,0x6f]
; CHECK: sri.2s	v0, v0, #5              ; encoding: [0x00,0x44,0x3b,0x2f]
; CHECK: sri.4s	v0, v0, #6              ; encoding: [0x00,0x44,0x3a,0x6f]
; CHECK: sri.2d	v0, v0, #7              ; encoding: [0x00,0x44,0x79,0x6f]
; CHECK: srshr.8b	v0, v0, #1      ; encoding: [0x00,0x24,0x0f,0x0f]
; CHECK: srshr.16b	v0, v0, #2      ; encoding: [0x00,0x24,0x0e,0x4f]
; CHECK: srshr.4h	v0, v0, #3      ; encoding: [0x00,0x24,0x1d,0x0f]
; CHECK: srshr.8h	v0, v0, #4      ; encoding: [0x00,0x24,0x1c,0x4f]
; CHECK: srshr.2s	v0, v0, #5      ; encoding: [0x00,0x24,0x3b,0x0f]
; CHECK: srshr.4s	v0, v0, #6      ; encoding: [0x00,0x24,0x3a,0x4f]
; CHECK: srshr.2d	v0, v0, #7      ; encoding: [0x00,0x24,0x79,0x4f]
; CHECK: srsra.8b	v0, v0, #1      ; encoding: [0x00,0x34,0x0f,0x0f]
; CHECK: srsra.16b	v0, v0, #2      ; encoding: [0x00,0x34,0x0e,0x4f]
; CHECK: srsra.4h	v0, v0, #3      ; encoding: [0x00,0x34,0x1d,0x0f]
; CHECK: srsra.8h	v0, v0, #4      ; encoding: [0x00,0x34,0x1c,0x4f]
; CHECK: srsra.2s	v0, v0, #5      ; encoding: [0x00,0x34,0x3b,0x0f]
; CHECK: srsra.4s	v0, v0, #6      ; encoding: [0x00,0x34,0x3a,0x4f]
; CHECK: srsra.2d	v0, v0, #7      ; encoding: [0x00,0x34,0x79,0x4f]
; CHECK: sshll.8h	v0, v0, #1      ; encoding: [0x00,0xa4,0x09,0x0f]
; CHECK: sshll2.8h	v0, v0, #2      ; encoding: [0x00,0xa4,0x0a,0x4f]
; CHECK: sshll.4s	v0, v0, #3      ; encoding: [0x00,0xa4,0x13,0x0f]
; CHECK: sshll2.4s	v0, v0, #4      ; encoding: [0x00,0xa4,0x14,0x4f]
; CHECK: sshll.2d	v0, v0, #5      ; encoding: [0x00,0xa4,0x25,0x0f]
; CHECK: sshll2.2d	v0, v0, #6      ; encoding: [0x00,0xa4,0x26,0x4f]
; CHECK: sshr.8b	v0, v0, #1              ; encoding: [0x00,0x04,0x0f,0x0f]
; CHECK: sshr.16b	v0, v0, #2      ; encoding: [0x00,0x04,0x0e,0x4f]
; CHECK: sshr.4h	v0, v0, #3              ; encoding: [0x00,0x04,0x1d,0x0f]
; CHECK: sshr.8h	v0, v0, #4              ; encoding: [0x00,0x04,0x1c,0x4f]
; CHECK: sshr.2s	v0, v0, #5              ; encoding: [0x00,0x04,0x3b,0x0f]
; CHECK: sshr.4s	v0, v0, #6              ; encoding: [0x00,0x04,0x3a,0x4f]
; CHECK: sshr.2d	v0, v0, #7              ; encoding: [0x00,0x04,0x79,0x4f]
; CHECK: sshr.8b	v0, v0, #1              ; encoding: [0x00,0x04,0x0f,0x0f]
; CHECK: ssra.16b	v0, v0, #2      ; encoding: [0x00,0x14,0x0e,0x4f]
; CHECK: ssra.4h	v0, v0, #3              ; encoding: [0x00,0x14,0x1d,0x0f]
; CHECK: ssra.8h	v0, v0, #4              ; encoding: [0x00,0x14,0x1c,0x4f]
; CHECK: ssra.2s	v0, v0, #5              ; encoding: [0x00,0x14,0x3b,0x0f]
; CHECK: ssra.4s	v0, v0, #6              ; encoding: [0x00,0x14,0x3a,0x4f]
; CHECK: ssra.2d	v0, v0, #7              ; encoding: [0x00,0x14,0x79,0x4f]
; CHECK: ssra		d0, d0, #64             ; encoding: [0x00,0x14,0x40,0x5f]
; CHECK: ucvtf.2s	v0, v0, #1      ; encoding: [0x00,0xe4,0x3f,0x2f]
; CHECK: ucvtf.4s	v0, v0, #2      ; encoding: [0x00,0xe4,0x3e,0x6f]
; CHECK: ucvtf.2d	v0, v0, #3      ; encoding: [0x00,0xe4,0x7d,0x6f]
; CHECK: uqrshrn.8b	v0, v0, #1      ; encoding: [0x00,0x9c,0x0f,0x2f]
; CHECK: uqrshrn2.16b	v0, v0, #2      ; encoding: [0x00,0x9c,0x0e,0x6f]
; CHECK: uqrshrn.4h	v0, v0, #3      ; encoding: [0x00,0x9c,0x1d,0x2f]
; CHECK: uqrshrn2.8h	v0, v0, #4      ; encoding: [0x00,0x9c,0x1c,0x6f]
; CHECK: uqrshrn.2s	v0, v0, #5      ; encoding: [0x00,0x9c,0x3b,0x2f]
; CHECK: uqrshrn2.4s	v0, v0, #6      ; encoding: [0x00,0x9c,0x3a,0x6f]
; CHECK: uqshl.8b	v0, v0, #1      ; encoding: [0x00,0x74,0x09,0x2f]
; CHECK: uqshl.16b	v0, v0, #2      ; encoding: [0x00,0x74,0x0a,0x6f]
; CHECK: uqshl.4h	v0, v0, #3      ; encoding: [0x00,0x74,0x13,0x2f]
; CHECK: uqshl.8h	v0, v0, #4      ; encoding: [0x00,0x74,0x14,0x6f]
; CHECK: uqshl.2s	v0, v0, #5      ; encoding: [0x00,0x74,0x25,0x2f]
; CHECK: uqshl.4s	v0, v0, #6      ; encoding: [0x00,0x74,0x26,0x6f]
; CHECK: uqshl.2d	v0, v0, #7      ; encoding: [0x00,0x74,0x47,0x6f]
; CHECK: uqshrn.8b	v0, v0, #1      ; encoding: [0x00,0x94,0x0f,0x2f]
; CHECK: uqshrn2.16b	v0, v0, #2      ; encoding: [0x00,0x94,0x0e,0x6f]
; CHECK: uqshrn.4h	v0, v0, #3      ; encoding: [0x00,0x94,0x1d,0x2f]
; CHECK: uqshrn2.8h	v0, v0, #4      ; encoding: [0x00,0x94,0x1c,0x6f]
; CHECK: uqshrn.2s	v0, v0, #5      ; encoding: [0x00,0x94,0x3b,0x2f]
; CHECK: uqshrn2.4s	v0, v0, #6      ; encoding: [0x00,0x94,0x3a,0x6f]
; CHECK: urshr.8b	v0, v0, #1      ; encoding: [0x00,0x24,0x0f,0x2f]
; CHECK: urshr.16b	v0, v0, #2      ; encoding: [0x00,0x24,0x0e,0x6f]
; CHECK: urshr.4h	v0, v0, #3      ; encoding: [0x00,0x24,0x1d,0x2f]
; CHECK: urshr.8h	v0, v0, #4      ; encoding: [0x00,0x24,0x1c,0x6f]
; CHECK: urshr.2s	v0, v0, #5      ; encoding: [0x00,0x24,0x3b,0x2f]
; CHECK: urshr.4s	v0, v0, #6      ; encoding: [0x00,0x24,0x3a,0x6f]
; CHECK: urshr.2d	v0, v0, #7      ; encoding: [0x00,0x24,0x79,0x6f]
; CHECK: ursra.8b	v0, v0, #1      ; encoding: [0x00,0x34,0x0f,0x2f]
; CHECK: ursra.16b	v0, v0, #2      ; encoding: [0x00,0x34,0x0e,0x6f]
; CHECK: ursra.4h	v0, v0, #3      ; encoding: [0x00,0x34,0x1d,0x2f]
; CHECK: ursra.8h	v0, v0, #4      ; encoding: [0x00,0x34,0x1c,0x6f]
; CHECK: ursra.2s	v0, v0, #5      ; encoding: [0x00,0x34,0x3b,0x2f]
; CHECK: ursra.4s	v0, v0, #6      ; encoding: [0x00,0x34,0x3a,0x6f]
; CHECK: ursra.2d	v0, v0, #7      ; encoding: [0x00,0x34,0x79,0x6f]
; CHECK: ushll.8h	v0, v0, #1      ; encoding: [0x00,0xa4,0x09,0x2f]
; CHECK: ushll2.8h	v0, v0, #2      ; encoding: [0x00,0xa4,0x0a,0x6f]
; CHECK: ushll.4s	v0, v0, #3      ; encoding: [0x00,0xa4,0x13,0x2f]
; CHECK: ushll2.4s	v0, v0, #4      ; encoding: [0x00,0xa4,0x14,0x6f]
; CHECK: ushll.2d	v0, v0, #5      ; encoding: [0x00,0xa4,0x25,0x2f]
; CHECK: ushll2.2d	v0, v0, #6      ; encoding: [0x00,0xa4,0x26,0x6f]
; CHECK: ushr.8b	v0, v0, #1              ; encoding: [0x00,0x04,0x0f,0x2f]
; CHECK: ushr.16b	v0, v0, #2      ; encoding: [0x00,0x04,0x0e,0x6f]
; CHECK: ushr.4h	v0, v0, #3              ; encoding: [0x00,0x04,0x1d,0x2f]
; CHECK: ushr.8h	v0, v0, #4              ; encoding: [0x00,0x04,0x1c,0x6f]
; CHECK: ushr.2s	v0, v0, #5              ; encoding: [0x00,0x04,0x3b,0x2f]
; CHECK: ushr.4s	v0, v0, #6              ; encoding: [0x00,0x04,0x3a,0x6f]
; CHECK: ushr.2d	v0, v0, #7              ; encoding: [0x00,0x04,0x79,0x6f]
; CHECK: usra.8b	v0, v0, #1              ; encoding: [0x00,0x14,0x0f,0x2f]
; CHECK: usra.16b	v0, v0, #2      ; encoding: [0x00,0x14,0x0e,0x6f]
; CHECK: usra.4h	v0, v0, #3              ; encoding: [0x00,0x14,0x1d,0x2f]
; CHECK: usra.8h	v0, v0, #4              ; encoding: [0x00,0x14,0x1c,0x6f]
; CHECK: usra.2s	v0, v0, #5              ; encoding: [0x00,0x14,0x3b,0x2f]
; CHECK: usra.4s	v0, v0, #6              ; encoding: [0x00,0x14,0x3a,0x6f]
; CHECK: usra.2d	v0, v0, #7              ; encoding: [0x00,0x14,0x79,0x6f]


; ARM Verbose syntax variants.

   rshrn v9.8b, v11.8h, #1
   rshrn2 v8.16b, v9.8h, #2
   rshrn v7.4h, v8.4s, #3
   rshrn2 v6.8h, v7.4s, #4
   rshrn v5.2s, v6.2d, #5
   rshrn2 v4.4s, v5.2d, #6

   shrn v9.8b, v11.8h, #1
   shrn2 v8.16b, v9.8h, #2
   shrn v7.4h, v8.4s, #3
   shrn2 v6.8h, v7.4s, #4
   shrn v5.2s, v6.2d, #5
   shrn2 v4.4s, v5.2d, #6

   sqrshrn v9.8b, v11.8h, #1
   sqrshrn2 v8.16b, v9.8h, #2
   sqrshrn v7.4h, v8.4s, #3
   sqrshrn2 v6.8h, v7.4s, #4
   sqrshrn v5.2s, v6.2d, #5
   sqrshrn2 v4.4s, v5.2d, #6

   sqshrn v9.8b, v11.8h, #1
   sqshrn2 v8.16b, v9.8h, #2
   sqshrn v7.4h, v8.4s, #3
   sqshrn2 v6.8h, v7.4s, #4
   sqshrn v5.2s, v6.2d, #5
   sqshrn2 v4.4s, v5.2d, #6

   sqrshrun v9.8b, v11.8h, #1
   sqrshrun2 v8.16b, v9.8h, #2
   sqrshrun v7.4h, v8.4s, #3
   sqrshrun2 v6.8h, v7.4s, #4
   sqrshrun v5.2s, v6.2d, #5
   sqrshrun2 v4.4s, v5.2d, #6

   sqshrun v9.8b, v11.8h, #1
   sqshrun2 v8.16b, v9.8h, #2
   sqshrun v7.4h, v8.4s, #3
   sqshrun2 v6.8h, v7.4s, #4
   sqshrun v5.2s, v6.2d, #5
   sqshrun2 v4.4s, v5.2d, #6

   uqrshrn v9.8b, v11.8h, #1
   uqrshrn2 v8.16b, v9.8h, #2
   uqrshrn v7.4h, v8.4s, #3
   uqrshrn2 v6.8h, v7.4s, #4
   uqrshrn v5.2s, v6.2d, #5
   uqrshrn2 v4.4s, v5.2d, #6

   uqshrn v9.8b, v11.8h, #1
   uqshrn2 v8.16b, v9.8h, #2
   uqshrn v7.4h, v8.4s, #3
   uqshrn2 v6.8h, v7.4s, #4
   uqshrn v5.2s, v6.2d, #5
   uqshrn2 v4.4s, v5.2d, #6

   sshll2 v10.8h, v3.16b, #6
   sshll2 v11.4s, v4.8h, #5
   sshll2 v12.2d, v5.4s, #4
   sshll v13.8h, v6.8b, #3
   sshll v14.4s, v7.4h, #2
   sshll v15.2d, v8.2s, #7

   ushll2 v10.8h, v3.16b, #6
   ushll2 v11.4s, v4.8h, #5
   ushll2 v12.2d, v5.4s, #4
   ushll v13.8h, v6.8b, #3
   ushll v14.4s, v7.4h, #2
   ushll v15.2d, v8.2s, #7


; CHECK: rshrn.8b	v9, v11, #1     ; encoding: [0x69,0x8d,0x0f,0x0f]
; CHECK: rshrn2.16b	v8, v9, #2      ; encoding: [0x28,0x8d,0x0e,0x4f]
; CHECK: rshrn.4h	v7, v8, #3      ; encoding: [0x07,0x8d,0x1d,0x0f]
; CHECK: rshrn2.8h	v6, v7, #4      ; encoding: [0xe6,0x8c,0x1c,0x4f]
; CHECK: rshrn.2s	v5, v6, #5      ; encoding: [0xc5,0x8c,0x3b,0x0f]
; CHECK: rshrn2.4s	v4, v5, #6      ; encoding: [0xa4,0x8c,0x3a,0x4f]
; CHECK: shrn.8b	v9, v11, #1             ; encoding: [0x69,0x85,0x0f,0x0f]
; CHECK: shrn2.16b	v8, v9, #2      ; encoding: [0x28,0x85,0x0e,0x4f]
; CHECK: shrn.4h	v7, v8, #3              ; encoding: [0x07,0x85,0x1d,0x0f]
; CHECK: shrn2.8h	v6, v7, #4      ; encoding: [0xe6,0x84,0x1c,0x4f]
; CHECK: shrn.2s	v5, v6, #5              ; encoding: [0xc5,0x84,0x3b,0x0f]
; CHECK: shrn2.4s	v4, v5, #6      ; encoding: [0xa4,0x84,0x3a,0x4f]
; CHECK: sqrshrn.8b	v9, v11, #1     ; encoding: [0x69,0x9d,0x0f,0x0f]
; CHECK: sqrshrn2.16b	v8, v9, #2      ; encoding: [0x28,0x9d,0x0e,0x4f]
; CHECK: sqrshrn.4h	v7, v8, #3      ; encoding: [0x07,0x9d,0x1d,0x0f]
; CHECK: sqrshrn2.8h	v6, v7, #4      ; encoding: [0xe6,0x9c,0x1c,0x4f]
; CHECK: sqrshrn.2s	v5, v6, #5      ; encoding: [0xc5,0x9c,0x3b,0x0f]
; CHECK: sqrshrn2.4s	v4, v5, #6      ; encoding: [0xa4,0x9c,0x3a,0x4f]
; CHECK: sqshrn.8b	v9, v11, #1     ; encoding: [0x69,0x95,0x0f,0x0f]
; CHECK: sqshrn2.16b	v8, v9, #2      ; encoding: [0x28,0x95,0x0e,0x4f]
; CHECK: sqshrn.4h	v7, v8, #3      ; encoding: [0x07,0x95,0x1d,0x0f]
; CHECK: sqshrn2.8h	v6, v7, #4      ; encoding: [0xe6,0x94,0x1c,0x4f]
; CHECK: sqshrn.2s	v5, v6, #5      ; encoding: [0xc5,0x94,0x3b,0x0f]
; CHECK: sqshrn2.4s	v4, v5, #6      ; encoding: [0xa4,0x94,0x3a,0x4f]
; CHECK: sqrshrun.8b	v9, v11, #1     ; encoding: [0x69,0x8d,0x0f,0x2f]
; CHECK: sqrshrun2.16b	v8, v9, #2      ; encoding: [0x28,0x8d,0x0e,0x6f]
; CHECK: sqrshrun.4h	v7, v8, #3      ; encoding: [0x07,0x8d,0x1d,0x2f]
; CHECK: sqrshrun2.8h	v6, v7, #4      ; encoding: [0xe6,0x8c,0x1c,0x6f]
; CHECK: sqrshrun.2s	v5, v6, #5      ; encoding: [0xc5,0x8c,0x3b,0x2f]
; CHECK: sqrshrun2.4s	v4, v5, #6      ; encoding: [0xa4,0x8c,0x3a,0x6f]
; CHECK: sqshrun.8b	v9, v11, #1     ; encoding: [0x69,0x85,0x0f,0x2f]
; CHECK: sqshrun2.16b	v8, v9, #2      ; encoding: [0x28,0x85,0x0e,0x6f]
; CHECK: sqshrun.4h	v7, v8, #3      ; encoding: [0x07,0x85,0x1d,0x2f]
; CHECK: sqshrun2.8h	v6, v7, #4      ; encoding: [0xe6,0x84,0x1c,0x6f]
; CHECK: sqshrun.2s	v5, v6, #5      ; encoding: [0xc5,0x84,0x3b,0x2f]
; CHECK: sqshrun2.4s	v4, v5, #6      ; encoding: [0xa4,0x84,0x3a,0x6f]
; CHECK: uqrshrn.8b	v9, v11, #1     ; encoding: [0x69,0x9d,0x0f,0x2f]
; CHECK: uqrshrn2.16b	v8, v9, #2      ; encoding: [0x28,0x9d,0x0e,0x6f]
; CHECK: uqrshrn.4h	v7, v8, #3      ; encoding: [0x07,0x9d,0x1d,0x2f]
; CHECK: uqrshrn2.8h	v6, v7, #4      ; encoding: [0xe6,0x9c,0x1c,0x6f]
; CHECK: uqrshrn.2s	v5, v6, #5      ; encoding: [0xc5,0x9c,0x3b,0x2f]
; CHECK: uqrshrn2.4s	v4, v5, #6      ; encoding: [0xa4,0x9c,0x3a,0x6f]
; CHECK: uqshrn.8b	v9, v11, #1     ; encoding: [0x69,0x95,0x0f,0x2f]
; CHECK: uqshrn2.16b	v8, v9, #2      ; encoding: [0x28,0x95,0x0e,0x6f]
; CHECK: uqshrn.4h	v7, v8, #3      ; encoding: [0x07,0x95,0x1d,0x2f]
; CHECK: uqshrn2.8h	v6, v7, #4      ; encoding: [0xe6,0x94,0x1c,0x6f]
; CHECK: uqshrn.2s	v5, v6, #5      ; encoding: [0xc5,0x94,0x3b,0x2f]
; CHECK: uqshrn2.4s	v4, v5, #6      ; encoding: [0xa4,0x94,0x3a,0x6f]
; CHECK: sshll2.8h	v10, v3, #6     ; encoding: [0x6a,0xa4,0x0e,0x4f]
; CHECK: sshll2.4s	v11, v4, #5     ; encoding: [0x8b,0xa4,0x15,0x4f]
; CHECK: sshll2.2d	v12, v5, #4     ; encoding: [0xac,0xa4,0x24,0x4f]
; CHECK: sshll.8h	v13, v6, #3     ; encoding: [0xcd,0xa4,0x0b,0x0f]
; CHECK: sshll.4s	v14, v7, #2     ; encoding: [0xee,0xa4,0x12,0x0f]
; CHECK: sshll.2d	v15, v8, #7     ; encoding: [0x0f,0xa5,0x27,0x0f]
; CHECK: ushll2.8h	v10, v3, #6     ; encoding: [0x6a,0xa4,0x0e,0x6f]
; CHECK: ushll2.4s	v11, v4, #5     ; encoding: [0x8b,0xa4,0x15,0x6f]
; CHECK: ushll2.2d	v12, v5, #4     ; encoding: [0xac,0xa4,0x24,0x6f]
; CHECK: ushll.8h	v13, v6, #3     ; encoding: [0xcd,0xa4,0x0b,0x2f]
; CHECK: ushll.4s	v14, v7, #2     ; encoding: [0xee,0xa4,0x12,0x2f]
; CHECK: ushll.2d	v15, v8, #7     ; encoding: [0x0f,0xa5,0x27,0x2f]


  pmull.8h v0, v0, v0
  pmull2.8h v0, v0, v0
  pmull.1q v2, v3, v4
  pmull2.1q v2, v3, v4
  pmull v2.1q, v3.1d, v4.1d
  pmull2 v2.1q, v3.2d, v4.2d

; CHECK: pmull.8h	v0, v0, v0      ; encoding: [0x00,0xe0,0x20,0x0e]
; CHECK: pmull2.8h	v0, v0, v0      ; encoding: [0x00,0xe0,0x20,0x4e]
; CHECK: pmull.1q	v2, v3, v4      ; encoding: [0x62,0xe0,0xe4,0x0e]
; CHECK: pmull2.1q	v2, v3, v4      ; encoding: [0x62,0xe0,0xe4,0x4e]
; CHECK: pmull.1q	v2, v3, v4      ; encoding: [0x62,0xe0,0xe4,0x0e]
; CHECK: pmull2.1q	v2, v3, v4      ; encoding: [0x62,0xe0,0xe4,0x4e]


  faddp.2d d1, v2
  faddp.2s s3, v4
; CHECK: faddp.2d	d1, v2          ; encoding: [0x41,0xd8,0x70,0x7e]
; CHECK: faddp.2s	s3, v4          ; encoding: [0x83,0xd8,0x30,0x7e]

  tbl.16b v2, {v4,v5,v6,v7}, v1
  tbl.8b v0, {v4,v5,v6,v7}, v1
  tbl.16b v2, {v5}, v1
  tbl.8b v0, {v5}, v1
  tbl.16b v2, {v5,v6,v7}, v1
  tbl.8b v0, {v5,v6,v7}, v1
  tbl.16b v2, {v6,v7}, v1
  tbl.8b v0, {v6,v7}, v1
; CHECK: tbl.16b	v2, { v4, v5, v6, v7 }, v1 ; encoding: [0x82,0x60,0x01,0x4e]
; CHECK: tbl.8b	v0, { v4, v5, v6, v7 }, v1 ; encoding: [0x80,0x60,0x01,0x0e]
; CHECK: tbl.16b	v2, { v5 }, v1          ; encoding: [0xa2,0x00,0x01,0x4e]
; CHECK: tbl.8b	v0, { v5 }, v1          ; encoding: [0xa0,0x00,0x01,0x0e]
; CHECK: tbl.16b	v2, { v5, v6, v7 }, v1  ; encoding: [0xa2,0x40,0x01,0x4e]
; CHECK: tbl.8b	v0, { v5, v6, v7 }, v1  ; encoding: [0xa0,0x40,0x01,0x0e]
; CHECK: tbl.16b	v2, { v6, v7 }, v1      ; encoding: [0xc2,0x20,0x01,0x4e]
; CHECK: tbl.8b	v0, { v6, v7 }, v1      ; encoding: [0xc0,0x20,0x01,0x0e]

  tbl v2.16b, {v4.16b,v5.16b,v6.16b,v7.16b}, v1.16b
  tbl v0.8b, {v4.16b,v5.16b,v6.16b,v7.16b}, v1.8b
  tbl v2.16b, {v5.16b}, v1.16b
  tbl v0.8b, {v5.16b}, v1.8b
  tbl v2.16b, {v5.16b,v6.16b,v7.16b}, v1.16b
  tbl v0.8b, {v5.16b,v6.16b,v7.16b}, v1.8b
  tbl v2.16b, {v6.16b,v7.16b}, v1.16b
  tbl v0.8b, {v6.16b,v7.16b}, v1.8b
; CHECK: tbl.16b v2, { v4, v5, v6, v7 }, v1 ; encoding: [0x82,0x60,0x01,0x4e]
; CHECK: tbl.8b v0, { v4, v5, v6, v7 }, v1 ; encoding: [0x80,0x60,0x01,0x0e]
; CHECK: tbl.16b v2, { v5 }, v1          ; encoding: [0xa2,0x00,0x01,0x4e]
; CHECK: tbl.8b v0, { v5 }, v1          ; encoding: [0xa0,0x00,0x01,0x0e]
; CHECK: tbl.16b v2, { v5, v6, v7 }, v1  ; encoding: [0xa2,0x40,0x01,0x4e]
; CHECK: tbl.8b v0, { v5, v6, v7 }, v1  ; encoding: [0xa0,0x40,0x01,0x0e]
; CHECK: tbl.16b v2, { v6, v7 }, v1      ; encoding: [0xc2,0x20,0x01,0x4e]
; CHECK: tbl.8b v0, { v6, v7 }, v1      ; encoding: [0xc0,0x20,0x01,0x0e]

  sqdmull	s0, h0, h0
  sqdmull	d0, s0, s0
; CHECK: sqdmull	s0, h0, h0              ; encoding: [0x00,0xd0,0x60,0x5e]
; CHECK: sqdmull	d0, s0, s0              ; encoding: [0x00,0xd0,0xa0,0x5e]

  frsqrte s0, s0
  frsqrte d0, d0
; CHECK: frsqrte s0, s0                  ; encoding: [0x00,0xd8,0xa1,0x7e]
; CHECK: frsqrte d0, d0                  ; encoding: [0x00,0xd8,0xe1,0x7e]

  mov.16b v0, v0
  mov.2s v0, v0
; CHECK: mov.16b v0, v0              ; encoding: [0x00,0x1c,0xa0,0x4e]
; CHECK: mov.8b v0, v0              ; encoding: [0x00,0x1c,0xa0,0x0e]


; uadalp/sadalp verbose mode aliases.
  uadalp v14.4h, v25.8b
  uadalp v15.8h, v24.16b
  uadalp v16.2s, v23.4h
  uadalp v17.4s, v22.8h
  uadalp v18.1d, v21.2s
  uadalp v19.2d, v20.4s

  sadalp v1.4h, v11.8b
  sadalp v2.8h, v12.16b
  sadalp v3.2s, v13.4h
  sadalp v4.4s, v14.8h
  sadalp v5.1d, v15.2s
  sadalp v6.2d, v16.4s

; CHECK: uadalp.4h	v14, v25        ; encoding: [0x2e,0x6b,0x20,0x2e]
; CHECK: uadalp.8h	v15, v24        ; encoding: [0x0f,0x6b,0x20,0x6e]
; CHECK: uadalp.2s	v16, v23        ; encoding: [0xf0,0x6a,0x60,0x2e]
; CHECK: uadalp.4s	v17, v22        ; encoding: [0xd1,0x6a,0x60,0x6e]
; CHECK: uadalp.1d	v18, v21        ; encoding: [0xb2,0x6a,0xa0,0x2e]
; CHECK: uadalp.2d	v19, v20        ; encoding: [0x93,0x6a,0xa0,0x6e]
; CHECK: sadalp.4h	v1, v11         ; encoding: [0x61,0x69,0x20,0x0e]
; CHECK: sadalp.8h	v2, v12         ; encoding: [0x82,0x69,0x20,0x4e]
; CHECK: sadalp.2s	v3, v13         ; encoding: [0xa3,0x69,0x60,0x0e]
; CHECK: sadalp.4s	v4, v14         ; encoding: [0xc4,0x69,0x60,0x4e]
; CHECK: sadalp.1d	v5, v15         ; encoding: [0xe5,0x69,0xa0,0x0e]
; CHECK: sadalp.2d	v6, v16         ; encoding: [0x06,0x6a,0xa0,0x4e]

; MVN is an alias for 'not'.
  mvn v1.8b, v4.8b
  mvn v19.16b, v17.16b
  mvn.8b v10, v6
  mvn.16b v11, v7

; CHECK: mvn.8b	v1, v4                  ; encoding: [0x81,0x58,0x20,0x2e]
; CHECK: mvn.16b	v19, v17                ; encoding: [0x33,0x5a,0x20,0x6e]
; CHECK: mvn.8b	v10, v6                 ; encoding: [0xca,0x58,0x20,0x2e]
; CHECK: mvn.16b	v11, v7                 ; encoding: [0xeb,0x58,0x20,0x6e]

; sqdmull verbose mode aliases
 sqdmull v10.4s, v12.4h, v12.4h
 sqdmull2 v10.4s, v13.8h, v13.8h
 sqdmull v10.2d, v13.2s, v13.2s
 sqdmull2 v10.2d, v13.4s, v13.4s
; CHECK: sqdmull.4s	v10, v12, v12   ; encoding: [0x8a,0xd1,0x6c,0x0e]
; CHECK: sqdmull2.4s	v10, v13, v13   ; encoding: [0xaa,0xd1,0x6d,0x4e]
; CHECK: sqdmull.2d	v10, v13, v13   ; encoding: [0xaa,0xd1,0xad,0x0e]
; CHECK: sqdmull2.2d	v10, v13, v13   ; encoding: [0xaa,0xd1,0xad,0x4e]

; xtn verbose mode aliases
 xtn v14.8b, v14.8h
 xtn2 v14.16b, v14.8h
 xtn v14.4h, v14.4s
 xtn2 v14.8h, v14.4s
 xtn v14.2s, v14.2d
 xtn2 v14.4s, v14.2d
; CHECK: xtn.8b v14, v14                ; encoding: [0xce,0x29,0x21,0x0e]
; CHECK: xtn2.16b v14, v14              ; encoding: [0xce,0x29,0x21,0x4e]
; CHECK: xtn.4h v14, v14                ; encoding: [0xce,0x29,0x61,0x0e]
; CHECK: xtn2.8h v14, v14               ; encoding: [0xce,0x29,0x61,0x4e]
; CHECK: xtn.2s v14, v14                ; encoding: [0xce,0x29,0xa1,0x0e]
; CHECK: xtn2.4s v14, v14               ; encoding: [0xce,0x29,0xa1,0x4e]

; uaddl verbose mode aliases
 uaddl v9.8h, v13.8b, v14.8b
 uaddl2 v9.8h, v13.16b, v14.16b
 uaddl v9.4s, v13.4h, v14.4h
 uaddl2 v9.4s, v13.8h, v14.8h
 uaddl v9.2d, v13.2s, v14.2s
 uaddl2 v9.2d, v13.4s, v14.4s
; CHECK: uaddl.8h	v9, v13, v14    ; encoding: [0xa9,0x01,0x2e,0x2e]
; CHECK: uaddl2.8h	v9, v13, v14    ; encoding: [0xa9,0x01,0x2e,0x6e]
; CHECK: uaddl.4s	v9, v13, v14    ; encoding: [0xa9,0x01,0x6e,0x2e]
; CHECK: uaddl2.4s	v9, v13, v14    ; encoding: [0xa9,0x01,0x6e,0x6e]
; CHECK: uaddl.2d	v9, v13, v14    ; encoding: [0xa9,0x01,0xae,0x2e]
; CHECK: uaddl2.2d	v9, v13, v14    ; encoding: [0xa9,0x01,0xae,0x6e]

; bit verbose mode aliases
 bit v9.16b, v10.16b, v10.16b
 bit v9.8b, v10.8b, v10.8b
; CHECK: bit.16b v9, v10, v10           ; encoding: [0x49,0x1d,0xaa,0x6e]
; CHECK: bit.8b v9, v10, v10            ; encoding: [0x49,0x1d,0xaa,0x2e]

; pmull verbose mode aliases
 pmull v8.8h, v8.8b, v8.8b
 pmull2 v8.8h, v8.16b, v8.16b
 pmull v8.1q, v8.1d, v8.1d
 pmull2 v8.1q, v8.2d, v8.2d
; CHECK: pmull.8h	v8, v8, v8      ; encoding: [0x08,0xe1,0x28,0x0e]
; CHECK: pmull2.8h	v8, v8, v8      ; encoding: [0x08,0xe1,0x28,0x4e]
; CHECK: pmull.1q	v8, v8, v8      ; encoding: [0x08,0xe1,0xe8,0x0e]
; CHECK: pmull2.1q	v8, v8, v8      ; encoding: [0x08,0xe1,0xe8,0x4e]

; usubl verbose mode aliases
 usubl v9.8h, v13.8b, v14.8b
 usubl2 v9.8h, v13.16b, v14.16b
 usubl v9.4s, v13.4h, v14.4h
 usubl2 v9.4s, v13.8h, v14.8h
 usubl v9.2d, v13.2s, v14.2s
 usubl2 v9.2d, v13.4s, v14.4s
; CHECK: usubl.8h	v9, v13, v14    ; encoding: [0xa9,0x21,0x2e,0x2e]
; CHECK: usubl2.8h	v9, v13, v14    ; encoding: [0xa9,0x21,0x2e,0x6e]
; CHECK: usubl.4s	v9, v13, v14    ; encoding: [0xa9,0x21,0x6e,0x2e]
; CHECK: usubl2.4s	v9, v13, v14    ; encoding: [0xa9,0x21,0x6e,0x6e]
; CHECK: usubl.2d	v9, v13, v14    ; encoding: [0xa9,0x21,0xae,0x2e]
; CHECK: usubl2.2d	v9, v13, v14    ; encoding: [0xa9,0x21,0xae,0x6e]

; uabdl verbose mode aliases
 uabdl v9.8h, v13.8b, v14.8b
 uabdl2 v9.8h, v13.16b, v14.16b
 uabdl v9.4s, v13.4h, v14.4h
 uabdl2 v9.4s, v13.8h, v14.8h
 uabdl v9.2d, v13.2s, v14.2s
 uabdl2 v9.2d, v13.4s, v14.4s
; CHECK: uabdl.8h	v9, v13, v14    ; encoding: [0xa9,0x71,0x2e,0x2e]
; CHECK: uabdl2.8h	v9, v13, v14    ; encoding: [0xa9,0x71,0x2e,0x6e]
; CHECK: uabdl.4s	v9, v13, v14    ; encoding: [0xa9,0x71,0x6e,0x2e]
; CHECK: uabdl2.4s	v9, v13, v14    ; encoding: [0xa9,0x71,0x6e,0x6e]
; CHECK: uabdl.2d	v9, v13, v14    ; encoding: [0xa9,0x71,0xae,0x2e]
; CHECK: uabdl2.2d	v9, v13, v14    ; encoding: [0xa9,0x71,0xae,0x6e]

; umull verbose mode aliases
 umull v9.8h, v13.8b, v14.8b
 umull2 v9.8h, v13.16b, v14.16b
 umull v9.4s, v13.4h, v14.4h
 umull2 v9.4s, v13.8h, v14.8h
 umull v9.2d, v13.2s, v14.2s
 umull2 v9.2d, v13.4s, v14.4s
; CHECK: umull.8h	v9, v13, v14    ; encoding: [0xa9,0xc1,0x2e,0x2e]
; CHECK: umull2.8h	v9, v13, v14    ; encoding: [0xa9,0xc1,0x2e,0x6e]
; CHECK: umull.4s	v9, v13, v14    ; encoding: [0xa9,0xc1,0x6e,0x2e]
; CHECK: umull2.4s	v9, v13, v14    ; encoding: [0xa9,0xc1,0x6e,0x6e]
; CHECK: umull.2d	v9, v13, v14    ; encoding: [0xa9,0xc1,0xae,0x2e]
; CHECK: umull2.2d	v9, v13, v14    ; encoding: [0xa9,0xc1,0xae,0x6e]

; smull verbose mode aliases
 smull v9.8h, v13.8b, v14.8b
 smull2 v9.8h, v13.16b, v14.16b
 smull v9.4s, v13.4h, v14.4h
 smull2 v9.4s, v13.8h, v14.8h
 smull v9.2d, v13.2s, v14.2s
 smull2 v9.2d, v13.4s, v14.4s
; CHECK: smull.8h	v9, v13, v14    ; encoding: [0xa9,0xc1,0x2e,0x0e]
; CHECK: smull2.8h	v9, v13, v14    ; encoding: [0xa9,0xc1,0x2e,0x4e]
; CHECK: smull.4s	v9, v13, v14    ; encoding: [0xa9,0xc1,0x6e,0x0e]
; CHECK: smull2.4s	v9, v13, v14    ; encoding: [0xa9,0xc1,0x6e,0x4e]
; CHECK: smull.2d	v9, v13, v14    ; encoding: [0xa9,0xc1,0xae,0x0e]
; CHECK: smull2.2d	v9, v13, v14    ; encoding: [0xa9,0xc1,0xae,0x4e]
