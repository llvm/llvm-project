; RUN: llvm-mc -triple arm64-apple-darwin -show-encoding < %s | FileCheck %s

foo:
;-----------------------------------------------------------------------------
; Indexed loads
;-----------------------------------------------------------------------------

  ldr    w5, [x4, #20]
  ldr    x4, [x3]
  ldr    x2, [sp, #32]
  ldr    b5, [sp, #1]
  ldr    h6, [sp, #2]
  ldr    s7, [sp, #4]
  ldr    d8, [sp, #8]
  ldr    q9, [sp, #16]
  ldrb   w4, [x3]
  ldrb   w5, [x4, #20]
  ldrb	 w2, [x3, _foo@pageoff]
  ldrb   w3, [x2, "+[Test method].var"@PAGEOFF]
  ldrsb  w9, [x3]
  ldrsb  x2, [sp, #128]
  ldrh   w2, [sp, #32]
  ldrsh  w3, [sp, #32]
  ldrsh  x5, [x9, #24]
  ldrsw  x9, [sp, #512]

  prfm   #5, [sp, #32]
  prfm   #31, [sp, #32]
  prfm   pldl1keep, [x2]
  prfm   pldl1strm, [x2]
  prfm   pldl2keep, [x2]
  prfm   pldl2strm, [x2]
  prfm   pldl3keep, [x2]
  prfm   pldl3strm, [x2]
  prfm   pstl1keep, [x2]
  prfm   pstl1strm, [x2]
  prfm   pstl2keep, [x2]
  prfm   pstl2strm, [x2]
  prfm   pstl3keep, [x2]
  prfm   pstl3strm, [x2]
  prfm  pstl3strm, [x4, x5, lsl #3]

; CHECK: ldr    w5, [x4, #20]           ; encoding: [0x85,0x14,0x40,0xb9]
; CHECK: ldr    x4, [x3]                ; encoding: [0x64,0x00,0x40,0xf9]
; CHECK: ldr    x2, [sp, #32]           ; encoding: [0xe2,0x13,0x40,0xf9]
; CHECK: ldr    b5, [sp, #1]            ; encoding: [0xe5,0x07,0x40,0x3d]
; CHECK: ldr    h6, [sp, #2]            ; encoding: [0xe6,0x07,0x40,0x7d]
; CHECK: ldr    s7, [sp, #4]            ; encoding: [0xe7,0x07,0x40,0xbd]
; CHECK: ldr    d8, [sp, #8]            ; encoding: [0xe8,0x07,0x40,0xfd]
; CHECK: ldr    q9, [sp, #16]           ; encoding: [0xe9,0x07,0xc0,0x3d]
; CHECK: ldrb   w4, [x3]                ; encoding: [0x64,0x00,0x40,0x39]
; CHECK: ldrb   w5, [x4, #20]           ; encoding: [0x85,0x50,0x40,0x39]
; CHECK: ldrb	w2, [x3, _foo@PAGEOFF]  ; encoding: [0x62,0bAAAAAA00,0b01AAAAAA,0x39]
; CHECK: ldrb	w3, [x2, "+[Test method].var"@PAGEOFF] ; encoding: [0x43,0bAAAAAA00,0b01AAAAAA,0x39]
; CHECK: ldrsb  w9, [x3]                ; encoding: [0x69,0x00,0xc0,0x39]
; CHECK: ldrsb  x2, [sp, #128]          ; encoding: [0xe2,0x03,0x82,0x39]
; CHECK: ldrh   w2, [sp, #32]           ; encoding: [0xe2,0x43,0x40,0x79]
; CHECK: ldrsh  w3, [sp, #32]           ; encoding: [0xe3,0x43,0xc0,0x79]
; CHECK: ldrsh  x5, [x9, #24]           ; encoding: [0x25,0x31,0x80,0x79]
; CHECK: ldrsw  x9, [sp, #512]          ; encoding: [0xe9,0x03,0x82,0xb9]
; CHECK: prfm   pldl3strm, [sp, #32]    ; encoding: [0xe5,0x13,0x80,0xf9]
; CHECK: prfm	#31, [sp, #32]          ; encoding: [0xff,0x13,0x80,0xf9]
; CHECK: prfm   pldl1keep, [x2]         ; encoding: [0x40,0x00,0x80,0xf9]
; CHECK: prfm   pldl1strm, [x2]         ; encoding: [0x41,0x00,0x80,0xf9]
; CHECK: prfm   pldl2keep, [x2]         ; encoding: [0x42,0x00,0x80,0xf9]
; CHECK: prfm   pldl2strm, [x2]         ; encoding: [0x43,0x00,0x80,0xf9]
; CHECK: prfm   pldl3keep, [x2]         ; encoding: [0x44,0x00,0x80,0xf9]
; CHECK: prfm   pldl3strm, [x2]         ; encoding: [0x45,0x00,0x80,0xf9]
; CHECK: prfm   pstl1keep, [x2]         ; encoding: [0x50,0x00,0x80,0xf9]
; CHECK: prfm   pstl1strm, [x2]         ; encoding: [0x51,0x00,0x80,0xf9]
; CHECK: prfm   pstl2keep, [x2]         ; encoding: [0x52,0x00,0x80,0xf9]
; CHECK: prfm   pstl2strm, [x2]         ; encoding: [0x53,0x00,0x80,0xf9]
; CHECK: prfm   pstl3keep, [x2]         ; encoding: [0x54,0x00,0x80,0xf9]
; CHECK: prfm   pstl3strm, [x2]         ; encoding: [0x55,0x00,0x80,0xf9]
; CHECK: prfm	pstl3strm, [x4, x5, lsl #3] ; encoding: [0x95,0x78,0xa5,0xf8]

;-----------------------------------------------------------------------------
; Indexed stores
;-----------------------------------------------------------------------------

  str   x4, [x3]
  str   x2, [sp, #32]
  str   w5, [x4, #20]
  str   b5, [sp, #1]
  str   h6, [sp, #2]
  str   s7, [sp, #4]
  str   d8, [sp, #8]
  str   q9, [sp, #16]
  strb  w4, [x3]
  strb  w5, [x4, #20]
  strh  w2, [sp, #32]

; CHECK: str   x4, [x3]                 ; encoding: [0x64,0x00,0x00,0xf9]
; CHECK: str   x2, [sp, #32]            ; encoding: [0xe2,0x13,0x00,0xf9]
; CHECK: str   w5, [x4, #20]            ; encoding: [0x85,0x14,0x00,0xb9]
; CHECK: str   b5, [sp, #1]             ; encoding: [0xe5,0x07,0x00,0x3d]
; CHECK: str   h6, [sp, #2]             ; encoding: [0xe6,0x07,0x00,0x7d]
; CHECK: str   s7, [sp, #4]             ; encoding: [0xe7,0x07,0x00,0xbd]
; CHECK: str   d8, [sp, #8]             ; encoding: [0xe8,0x07,0x00,0xfd]
; CHECK: str   q9, [sp, #16]            ; encoding: [0xe9,0x07,0x80,0x3d]
; CHECK: strb  w4, [x3]                 ; encoding: [0x64,0x00,0x00,0x39]
; CHECK: strb  w5, [x4, #20]            ; encoding: [0x85,0x50,0x00,0x39]
; CHECK: strh  w2, [sp, #32]            ; encoding: [0xe2,0x43,0x00,0x79]

;-----------------------------------------------------------------------------
; Unscaled immediate loads and stores
;-----------------------------------------------------------------------------

  ldur    w2, [x3]
  ldur    w2, [sp, #24]
  ldur    x2, [x3]
  ldur    x2, [sp, #24]
  ldur    b5, [sp, #1]
  ldur    h6, [sp, #2]
  ldur    s7, [sp, #4]
  ldur    d8, [sp, #8]
  ldur    q9, [sp, #16]
  ldursb  w9, [x3]
  ldursb  x2, [sp, #128]
  ldursh  w3, [sp, #32]
  ldursh  x5, [x9, #24]
  ldursw  x9, [sp, #-128]

; CHECK: ldur    w2, [x3]               ; encoding: [0x62,0x00,0x40,0xb8]
; CHECK: ldur    w2, [sp, #24]          ; encoding: [0xe2,0x83,0x41,0xb8]
; CHECK: ldur    x2, [x3]               ; encoding: [0x62,0x00,0x40,0xf8]
; CHECK: ldur    x2, [sp, #24]          ; encoding: [0xe2,0x83,0x41,0xf8]
; CHECK: ldur    b5, [sp, #1]           ; encoding: [0xe5,0x13,0x40,0x3c]
; CHECK: ldur    h6, [sp, #2]           ; encoding: [0xe6,0x23,0x40,0x7c]
; CHECK: ldur    s7, [sp, #4]           ; encoding: [0xe7,0x43,0x40,0xbc]
; CHECK: ldur    d8, [sp, #8]           ; encoding: [0xe8,0x83,0x40,0xfc]
; CHECK: ldur    q9, [sp, #16]          ; encoding: [0xe9,0x03,0xc1,0x3c]
; CHECK: ldursb  w9, [x3]               ; encoding: [0x69,0x00,0xc0,0x38]
; CHECK: ldursb  x2, [sp, #128]         ; encoding: [0xe2,0x03,0x88,0x38]
; CHECK: ldursh  w3, [sp, #32]          ; encoding: [0xe3,0x03,0xc2,0x78]
; CHECK: ldursh  x5, [x9, #24]          ; encoding: [0x25,0x81,0x81,0x78]
; CHECK: ldursw  x9, [sp, #-128]        ; encoding: [0xe9,0x03,0x98,0xb8]

  stur    w4, [x3]
  stur    w2, [sp, #32]
  stur    x4, [x3]
  stur    x2, [sp, #32]
  stur    w5, [x4, #20]
  stur    b5, [sp, #1]
  stur    h6, [sp, #2]
  stur    s7, [sp, #4]
  stur    d8, [sp, #8]
  stur    q9, [sp, #16]
  sturb   w4, [x3]
  sturb   w5, [x4, #20]
  sturh   w2, [sp, #32]
  prfum   #5, [sp, #32]

; CHECK: stur    w4, [x3]               ; encoding: [0x64,0x00,0x00,0xb8]
; CHECK: stur    w2, [sp, #32]          ; encoding: [0xe2,0x03,0x02,0xb8]
; CHECK: stur    x4, [x3]               ; encoding: [0x64,0x00,0x00,0xf8]
; CHECK: stur    x2, [sp, #32]          ; encoding: [0xe2,0x03,0x02,0xf8]
; CHECK: stur    w5, [x4, #20]          ; encoding: [0x85,0x40,0x01,0xb8]
; CHECK: stur    b5, [sp, #1]           ; encoding: [0xe5,0x13,0x00,0x3c]
; CHECK: stur    h6, [sp, #2]           ; encoding: [0xe6,0x23,0x00,0x7c]
; CHECK: stur    s7, [sp, #4]           ; encoding: [0xe7,0x43,0x00,0xbc]
; CHECK: stur    d8, [sp, #8]           ; encoding: [0xe8,0x83,0x00,0xfc]
; CHECK: stur    q9, [sp, #16]          ; encoding: [0xe9,0x03,0x81,0x3c]
; CHECK: sturb   w4, [x3]               ; encoding: [0x64,0x00,0x00,0x38]
; CHECK: sturb   w5, [x4, #20]          ; encoding: [0x85,0x40,0x01,0x38]
; CHECK: sturh   w2, [sp, #32]          ; encoding: [0xe2,0x03,0x02,0x78]
; CHECK: prfum   pldl3strm, [sp, #32]   ; encoding: [0xe5,0x03,0x82,0xf8]

;-----------------------------------------------------------------------------
; Unprivileged loads and stores
;-----------------------------------------------------------------------------

  ldtr    w3, [x4, #16]
  ldtr    x3, [x4, #16]
  ldtrb   w3, [x4, #16]
  ldtrsb  w9, [x3]
  ldtrsb  x2, [sp, #128]
  ldtrh   w3, [x4, #16]
  ldtrsh  w3, [sp, #32]
  ldtrsh  x5, [x9, #24]
  ldtrsw  x9, [sp, #-128]

; CHECK: ldtr   w3, [x4, #16]           ; encoding: [0x83,0x08,0x41,0xb8]
; CHECK: ldtr   x3, [x4, #16]           ; encoding: [0x83,0x08,0x41,0xf8]
; CHECK: ldtrb  w3, [x4, #16]           ; encoding: [0x83,0x08,0x41,0x38]
; CHECK: ldtrsb w9, [x3]                ; encoding: [0x69,0x08,0xc0,0x38]
; CHECK: ldtrsb x2, [sp, #128]          ; encoding: [0xe2,0x0b,0x88,0x38]
; CHECK: ldtrh  w3, [x4, #16]           ; encoding: [0x83,0x08,0x41,0x78]
; CHECK: ldtrsh w3, [sp, #32]           ; encoding: [0xe3,0x0b,0xc2,0x78]
; CHECK: ldtrsh x5, [x9, #24]           ; encoding: [0x25,0x89,0x81,0x78]
; CHECK: ldtrsw x9, [sp, #-128]         ; encoding: [0xe9,0x0b,0x98,0xb8]

  sttr    w5, [x4, #20]
  sttr    x4, [x3]
  sttr    x2, [sp, #32]
  sttrb   w4, [x3]
  sttrb   w5, [x4, #20]
  sttrh   w2, [sp, #32]

; CHECK: sttr   w5, [x4, #20]           ; encoding: [0x85,0x48,0x01,0xb8]
; CHECK: sttr   x4, [x3]                ; encoding: [0x64,0x08,0x00,0xf8]
; CHECK: sttr   x2, [sp, #32]           ; encoding: [0xe2,0x0b,0x02,0xf8]
; CHECK: sttrb  w4, [x3]                ; encoding: [0x64,0x08,0x00,0x38]
; CHECK: sttrb  w5, [x4, #20]           ; encoding: [0x85,0x48,0x01,0x38]
; CHECK: sttrh  w2, [sp, #32]           ; encoding: [0xe2,0x0b,0x02,0x78]

;-----------------------------------------------------------------------------
; Pre-indexed loads and stores
;-----------------------------------------------------------------------------

  ldr   x29, [x7, #8]!
  ldr   x30, [x7, #8]!
  ldr   b5, [x0, #1]!
  ldr   h6, [x0, #2]!
  ldr   s7, [x0, #4]!
  ldr   d8, [x0, #8]!
  ldr   q9, [x0, #16]!

  str   x30, [x7, #-8]!
  str   x29, [x7, #-8]!
  str   b5, [x0, #-1]!
  str   h6, [x0, #-2]!
  str   s7, [x0, #-4]!
  str   d8, [x0, #-8]!
  str   q9, [x0, #-16]!

; CHECK: ldr  x29, [x7, #8]!             ; encoding: [0xfd,0x8c,0x40,0xf8]
; CHECK: ldr  x30, [x7, #8]!             ; encoding: [0xfe,0x8c,0x40,0xf8]
; CHECK: ldr  b5, [x0, #1]!             ; encoding: [0x05,0x1c,0x40,0x3c]
; CHECK: ldr  h6, [x0, #2]!             ; encoding: [0x06,0x2c,0x40,0x7c]
; CHECK: ldr  s7, [x0, #4]!             ; encoding: [0x07,0x4c,0x40,0xbc]
; CHECK: ldr  d8, [x0, #8]!             ; encoding: [0x08,0x8c,0x40,0xfc]
; CHECK: ldr  q9, [x0, #16]!            ; encoding: [0x09,0x0c,0xc1,0x3c]

; CHECK: str  x30, [x7, #-8]!            ; encoding: [0xfe,0x8c,0x1f,0xf8]
; CHECK: str  x29, [x7, #-8]!            ; encoding: [0xfd,0x8c,0x1f,0xf8]
; CHECK: str  b5, [x0, #-1]!            ; encoding: [0x05,0xfc,0x1f,0x3c]
; CHECK: str  h6, [x0, #-2]!            ; encoding: [0x06,0xec,0x1f,0x7c]
; CHECK: str  s7, [x0, #-4]!            ; encoding: [0x07,0xcc,0x1f,0xbc]
; CHECK: str  d8, [x0, #-8]!            ; encoding: [0x08,0x8c,0x1f,0xfc]
; CHECK: str  q9, [x0, #-16]!           ; encoding: [0x09,0x0c,0x9f,0x3c]

;-----------------------------------------------------------------------------
; post-indexed loads and stores
;-----------------------------------------------------------------------------
  str x30, [x7], #-8
  str x29, [x7], #-8
  str b5, [x0], #-1
  str h6, [x0], #-2
  str s7, [x0], #-4
  str d8, [x0], #-8
  str q9, [x0], #-16

  ldr x29, [x7], #8
  ldr x30, [x7], #8
  ldr b5, [x0], #1
  ldr h6, [x0], #2
  ldr s7, [x0], #4
  ldr d8, [x0], #8
  ldr q9, [x0], #16

; CHECK: str x30, [x7], #-8             ; encoding: [0xfe,0x84,0x1f,0xf8]
; CHECK: str x29, [x7], #-8             ; encoding: [0xfd,0x84,0x1f,0xf8]
; CHECK: str b5, [x0], #-1             ; encoding: [0x05,0xf4,0x1f,0x3c]
; CHECK: str h6, [x0], #-2             ; encoding: [0x06,0xe4,0x1f,0x7c]
; CHECK: str s7, [x0], #-4             ; encoding: [0x07,0xc4,0x1f,0xbc]
; CHECK: str d8, [x0], #-8             ; encoding: [0x08,0x84,0x1f,0xfc]
; CHECK: str q9, [x0], #-16            ; encoding: [0x09,0x04,0x9f,0x3c]

; CHECK: ldr x29, [x7], #8              ; encoding: [0xfd,0x84,0x40,0xf8]
; CHECK: ldr x30, [x7], #8              ; encoding: [0xfe,0x84,0x40,0xf8]
; CHECK: ldr b5, [x0], #1              ; encoding: [0x05,0x14,0x40,0x3c]
; CHECK: ldr h6, [x0], #2              ; encoding: [0x06,0x24,0x40,0x7c]
; CHECK: ldr s7, [x0], #4              ; encoding: [0x07,0x44,0x40,0xbc]
; CHECK: ldr d8, [x0], #8              ; encoding: [0x08,0x84,0x40,0xfc]
; CHECK: ldr q9, [x0], #16             ; encoding: [0x09,0x04,0xc1,0x3c]

;-----------------------------------------------------------------------------
; Load/Store pair (indexed, offset)
;-----------------------------------------------------------------------------

  ldp    w3, w2, [x15, #16]
  ldp    x4, x9, [sp, #-16]
  ldpsw  x2, x3, [x14, #16]
  ldpsw  x2, x3, [sp, #-16]
  ldp    s10, s1, [x2, #64]
  ldp    d10, d1, [x2]
  ldp    q2, q3, [x0, #32]

; CHECK: ldp    w3, w2, [x15, #16]      ; encoding: [0xe3,0x09,0x42,0x29]
; CHECK: ldp    x4, x9, [sp, #-16]      ; encoding: [0xe4,0x27,0x7f,0xa9]
; CHECK: ldpsw  x2, x3, [x14, #16]      ; encoding: [0xc2,0x0d,0x42,0x69]
; CHECK: ldpsw  x2, x3, [sp, #-16]      ; encoding: [0xe2,0x0f,0x7e,0x69]
; CHECK: ldp    s10, s1, [x2, #64]      ; encoding: [0x4a,0x04,0x48,0x2d]
; CHECK: ldp    d10, d1, [x2]           ; encoding: [0x4a,0x04,0x40,0x6d]
; CHECK: ldp    q2, q3, [x0, #32]       ; encoding: [0x02,0x0c,0x41,0xad]

  stp    w3, w2, [x15, #16]
  stp    x4, x9, [sp, #-16]
  stp    s10, s1, [x2, #64]
  stp    d10, d1, [x2]
  stp    q2, q3, [x0, #32]

; CHECK: stp    w3, w2, [x15, #16]      ; encoding: [0xe3,0x09,0x02,0x29]
; CHECK: stp    x4, x9, [sp, #-16]      ; encoding: [0xe4,0x27,0x3f,0xa9]
; CHECK: stp    s10, s1, [x2, #64]      ; encoding: [0x4a,0x04,0x08,0x2d]
; CHECK: stp    d10, d1, [x2]           ; encoding: [0x4a,0x04,0x00,0x6d]
; CHECK: stp    q2, q3, [x0, #32]       ; encoding: [0x02,0x0c,0x01,0xad]

;-----------------------------------------------------------------------------
; Load/Store pair (pre-indexed)
;-----------------------------------------------------------------------------

  ldp    w3, w2, [x15, #16]!
  ldp    x4, x9, [sp, #-16]!
  ldpsw  x2, x3, [x14, #16]!
  ldpsw  x2, x3, [sp, #-16]!
  ldp    s10, s1, [x2, #64]!
  ldp    d10, d1, [x2, #16]!

; CHECK: ldp  w3, w2, [x15, #16]!       ; encoding: [0xe3,0x09,0xc2,0x29]
; CHECK: ldp  x4, x9, [sp, #-16]!       ; encoding: [0xe4,0x27,0xff,0xa9]
; CHECK: ldpsw	x2, x3, [x14, #16]!     ; encoding: [0xc2,0x0d,0xc2,0x69]
; CHECK: ldpsw	x2, x3, [sp, #-16]!     ; encoding: [0xe2,0x0f,0xfe,0x69]
; CHECK: ldp  s10, s1, [x2, #64]!       ; encoding: [0x4a,0x04,0xc8,0x2d]
; CHECK: ldp  d10, d1, [x2, #16]!       ; encoding: [0x4a,0x04,0xc1,0x6d]

  stp    w3, w2, [x15, #16]!
  stp    x4, x9, [sp, #-16]!
  stp    s10, s1, [x2, #64]!
  stp    d10, d1, [x2, #16]!

; CHECK: stp  w3, w2, [x15, #16]!       ; encoding: [0xe3,0x09,0x82,0x29]
; CHECK: stp  x4, x9, [sp, #-16]!       ; encoding: [0xe4,0x27,0xbf,0xa9]
; CHECK: stp  s10, s1, [x2, #64]!       ; encoding: [0x4a,0x04,0x88,0x2d]
; CHECK: stp  d10, d1, [x2, #16]!       ; encoding: [0x4a,0x04,0x81,0x6d]

;-----------------------------------------------------------------------------
; Load/Store pair (post-indexed)
;-----------------------------------------------------------------------------

  ldp    w3, w2, [x15], #16
  ldp    x4, x9, [sp], #-16
  ldpsw  x2, x3, [x14], #16
  ldpsw  x2, x3, [sp], #-16
  ldp    s10, s1, [x2], #64
  ldp    d10, d1, [x2], #16

; CHECK: ldp  w3, w2, [x15], #16        ; encoding: [0xe3,0x09,0xc2,0x28]
; CHECK: ldp  x4, x9, [sp], #-16        ; encoding: [0xe4,0x27,0xff,0xa8]
; CHECK: ldpsw	x2, x3, [x14], #16      ; encoding: [0xc2,0x0d,0xc2,0x68]
; CHECK: ldpsw	x2, x3, [sp], #-16      ; encoding: [0xe2,0x0f,0xfe,0x68]
; CHECK: ldp  s10, s1, [x2], #64        ; encoding: [0x4a,0x04,0xc8,0x2c]
; CHECK: ldp  d10, d1, [x2], #16        ; encoding: [0x4a,0x04,0xc1,0x6c]

  stp    w3, w2, [x15], #16
  stp    x4, x9, [sp], #-16
  stp    s10, s1, [x2], #64
  stp    d10, d1, [x2], #16

; CHECK: stp  w3, w2, [x15], #16        ; encoding: [0xe3,0x09,0x82,0x28]
; CHECK: stp  x4, x9, [sp], #-16        ; encoding: [0xe4,0x27,0xbf,0xa8]
; CHECK: stp  s10, s1, [x2], #64        ; encoding: [0x4a,0x04,0x88,0x2c]
; CHECK: stp  d10, d1, [x2], #16        ; encoding: [0x4a,0x04,0x81,0x6c]

;-----------------------------------------------------------------------------
; Load/Store pair (no-allocate)
;-----------------------------------------------------------------------------

  ldnp  w3, w2, [x15, #16]
  ldnp  x4, x9, [sp, #-16]
  ldnp  s10, s1, [x2, #64]
  ldnp  d10, d1, [x2]

; CHECK: ldnp  w3, w2, [x15, #16]       ; encoding: [0xe3,0x09,0x42,0x28]
; CHECK: ldnp  x4, x9, [sp, #-16]       ; encoding: [0xe4,0x27,0x7f,0xa8]
; CHECK: ldnp  s10, s1, [x2, #64]       ; encoding: [0x4a,0x04,0x48,0x2c]
; CHECK: ldnp  d10, d1, [x2]            ; encoding: [0x4a,0x04,0x40,0x6c]

  stnp  w3, w2, [x15, #16]
  stnp  x4, x9, [sp, #-16]
  stnp  s10, s1, [x2, #64]
  stnp  d10, d1, [x2]

; CHECK: stnp  w3, w2, [x15, #16]       ; encoding: [0xe3,0x09,0x02,0x28]
; CHECK: stnp  x4, x9, [sp, #-16]       ; encoding: [0xe4,0x27,0x3f,0xa8]
; CHECK: stnp  s10, s1, [x2, #64]       ; encoding: [0x4a,0x04,0x08,0x2c]
; CHECK: stnp  d10, d1, [x2]            ; encoding: [0x4a,0x04,0x00,0x6c]

;-----------------------------------------------------------------------------
; Load/Store register offset
;-----------------------------------------------------------------------------

  ldr  w0, [x0, x0]
  ldr  w0, [x0, x0, lsl #2]
  ldr  x0, [x0, x0]
  ldr  x0, [x0, x0, lsl #3]
  ldr  x0, [x0, x0, sxtx]

; CHECK: ldr  w0, [x0, x0]              ; encoding: [0x00,0x68,0x60,0xb8]
; CHECK: ldr  w0, [x0, x0, lsl #2]      ; encoding: [0x00,0x78,0x60,0xb8]
; CHECK: ldr  x0, [x0, x0]              ; encoding: [0x00,0x68,0x60,0xf8]
; CHECK: ldr  x0, [x0, x0, lsl #3]      ; encoding: [0x00,0x78,0x60,0xf8]
; CHECK: ldr  x0, [x0, x0, sxtx]        ; encoding: [0x00,0xe8,0x60,0xf8]

  ldr  b1, [x1, x2]
  ldr  b1, [x1, x2, lsl #0]
  ldr  h1, [x1, x2]
  ldr  h1, [x1, x2, lsl #1]
  ldr  s1, [x1, x2]
  ldr  s1, [x1, x2, lsl #2]
  ldr  d1, [x1, x2]
  ldr  d1, [x1, x2, lsl #3]
  ldr  q1, [x1, x2]
  ldr  q1, [x1, x2, lsl #4]

; CHECK: ldr  b1, [x1, x2]              ; encoding: [0x21,0x68,0x62,0x3c]
; CHECK: ldr  b1, [x1, x2, lsl #0]      ; encoding: [0x21,0x78,0x62,0x3c]
; CHECK: ldr  h1, [x1, x2]              ; encoding: [0x21,0x68,0x62,0x7c]
; CHECK: ldr  h1, [x1, x2, lsl #1]      ; encoding: [0x21,0x78,0x62,0x7c]
; CHECK: ldr  s1, [x1, x2]              ; encoding: [0x21,0x68,0x62,0xbc]
; CHECK: ldr  s1, [x1, x2, lsl #2]      ; encoding: [0x21,0x78,0x62,0xbc]
; CHECK: ldr  d1, [x1, x2]              ; encoding: [0x21,0x68,0x62,0xfc]
; CHECK: ldr  d1, [x1, x2, lsl #3]      ; encoding: [0x21,0x78,0x62,0xfc]
; CHECK: ldr  q1, [x1, x2]              ; encoding: [0x21,0x68,0xe2,0x3c]
; CHECK: ldr  q1, [x1, x2, lsl #4]      ; encoding: [0x21,0x78,0xe2,0x3c]

  str  d1, [sp, x3]
  str  d1, [sp, w3, uxtw #3]
  str  q1, [sp, x3]
  str  q1, [sp, w3, uxtw #4]

; CHECK: str  d1, [sp, x3]              ; encoding: [0xe1,0x6b,0x23,0xfc]
; CHECK: str  d1, [sp, w3, uxtw #3]     ; encoding: [0xe1,0x5b,0x23,0xfc]
; CHECK: str  q1, [sp, x3]              ; encoding: [0xe1,0x6b,0xa3,0x3c]
; CHECK: str  q1, [sp, w3, uxtw #4]     ; encoding: [0xe1,0x5b,0xa3,0x3c]

;-----------------------------------------------------------------------------
; Load literal
;-----------------------------------------------------------------------------

  ldr    w5, foo
  ldr    x4, foo
  ldrsw  x9, foo
  prfm   #5, foo

; CHECK: ldr    w5, foo                 ; encoding: [0bAAA00101,A,A,0x18]
; CHECK: ldr    x4, foo                 ; encoding: [0bAAA00100,A,A,0x58]
; CHECK: ldrsw  x9, foo                 ; encoding: [0bAAA01001,A,A,0x98]
; CHECK: prfm   pldl3strm, foo          ; encoding: [0bAAA00101,A,A,0xd8]

;-----------------------------------------------------------------------------
; Load/Store exclusive
;-----------------------------------------------------------------------------

  ldxr   w6, [x1]
  ldxr   x6, [x1]
  ldxrb  w6, [x1]
  ldxrh  w6, [x1]
  ldxp   w7, w3, [x9]
  ldxp   x7, x3, [x9]

; CHECK: ldxrb  w6, [x1]                ; encoding: [0x26,0x7c,0x5f,0x08]
; CHECK: ldxrh  w6, [x1]                ; encoding: [0x26,0x7c,0x5f,0x48]
; CHECK: ldxp   w7, w3, [x9]            ; encoding: [0x27,0x0d,0x7f,0x88]
; CHECK: ldxp   x7, x3, [x9]            ; encoding: [0x27,0x0d,0x7f,0xc8]

  stxr   w1, x4, [x3]
  stxr   w1, w4, [x3]
  stxrb  w1, w4, [x3]
  stxrh  w1, w4, [x3]
  stxp   w1, x2, x6, [x1]
  stxp   w1, w2, w6, [x1]

; CHECK: stxr   w1, x4, [x3]            ; encoding: [0x64,0x7c,0x01,0xc8]
; CHECK: stxr   w1, w4, [x3]            ; encoding: [0x64,0x7c,0x01,0x88]
; CHECK: stxrb  w1, w4, [x3]            ; encoding: [0x64,0x7c,0x01,0x08]
; CHECK: stxrh  w1, w4, [x3]            ; encoding: [0x64,0x7c,0x01,0x48]
; CHECK: stxp   w1, x2, x6, [x1]        ; encoding: [0x22,0x18,0x21,0xc8]
; CHECK: stxp   w1, w2, w6, [x1]        ; encoding: [0x22,0x18,0x21,0x88]

;-----------------------------------------------------------------------------
; Load-acquire/Store-release non-exclusive
;-----------------------------------------------------------------------------

  ldar   w4, [sp]
  ldar   x4, [sp, #0]
  ldarb  w4, [sp]
  ldarh  w4, [sp]

; CHECK: ldar   w4, [sp]                ; encoding: [0xe4,0xff,0xdf,0x88]
; CHECK: ldar   x4, [sp]                ; encoding: [0xe4,0xff,0xdf,0xc8]
; CHECK: ldarb  w4, [sp]                ; encoding: [0xe4,0xff,0xdf,0x08]
; CHECK: ldarh  w4, [sp]                ; encoding: [0xe4,0xff,0xdf,0x48]

  stlr   w3, [x6]
  stlr   x3, [x6]
  stlrb  w3, [x6]
  stlrh  w3, [x6]

; CHECK: stlr   w3, [x6]                ; encoding: [0xc3,0xfc,0x9f,0x88]
; CHECK: stlr   x3, [x6]                ; encoding: [0xc3,0xfc,0x9f,0xc8]
; CHECK: stlrb  w3, [x6]                ; encoding: [0xc3,0xfc,0x9f,0x08]
; CHECK: stlrh  w3, [x6]                ; encoding: [0xc3,0xfc,0x9f,0x48]

;-----------------------------------------------------------------------------
; Load-acquire/Store-release exclusive
;-----------------------------------------------------------------------------

  ldaxr   w2, [x4]
  ldaxr   x2, [x4]
  ldaxrb  w2, [x4, #0]
  ldaxrh  w2, [x4]
  ldaxp   w2, w6, [x1]
  ldaxp   x2, x6, [x1]

; CHECK: ldaxr   w2, [x4]               ; encoding: [0x82,0xfc,0x5f,0x88]
; CHECK: ldaxr   x2, [x4]               ; encoding: [0x82,0xfc,0x5f,0xc8]
; CHECK: ldaxrb  w2, [x4]               ; encoding: [0x82,0xfc,0x5f,0x08]
; CHECK: ldaxrh  w2, [x4]               ; encoding: [0x82,0xfc,0x5f,0x48]
; CHECK: ldaxp   w2, w6, [x1]           ; encoding: [0x22,0x98,0x7f,0x88]
; CHECK: ldaxp   x2, x6, [x1]           ; encoding: [0x22,0x98,0x7f,0xc8]

  stlxr   w8, x7, [x1]
  stlxr   w8, w7, [x1]
  stlxrb  w8, w7, [x1]
  stlxrh  w8, w7, [x1]
  stlxp   w1, x2, x6, [x1]
  stlxp   w1, w2, w6, [x1]

; CHECK: stlxr  w8, x7, [x1]            ; encoding: [0x27,0xfc,0x08,0xc8]
; CHECK: stlxr  w8, w7, [x1]            ; encoding: [0x27,0xfc,0x08,0x88]
; CHECK: stlxrb w8, w7, [x1]            ; encoding: [0x27,0xfc,0x08,0x08]
; CHECK: stlxrh w8, w7, [x1]            ; encoding: [0x27,0xfc,0x08,0x48]
; CHECK: stlxp  w1, x2, x6, [x1]        ; encoding: [0x22,0x98,0x21,0xc8]
; CHECK: stlxp  w1, w2, w6, [x1]        ; encoding: [0x22,0x98,0x21,0x88]


;-----------------------------------------------------------------------------
; LDUR/STUR aliases for negative and unaligned LDR/STR instructions.
;
; According to the ARM ISA documentation:
; "A programmer-friendly assembler should also generate these instructions
; in response to the standard LDR/STR mnemonics when the immediate offset is
; unambiguous, i.e. negative or unaligned."
;-----------------------------------------------------------------------------

  ldr x11, [x29, #-8]
  ldr x11, [x29, #7]
  ldr w0, [x0, #2]
  ldr w0, [x0, #-256]
  ldr b2, [x1, #-2]
  ldr h3, [x2, #3]
  ldr h3, [x3, #-4]
  ldr s3, [x4, #3]
  ldr s3, [x5, #-4]
  ldr d4, [x6, #4]
  ldr d4, [x7, #-8]
  ldr q5, [x8, #8]
  ldr q5, [x9, #-16]

; CHECK: ldur	x11, [x29, #-8]          ; encoding: [0xab,0x83,0x5f,0xf8]
; CHECK: ldur	x11, [x29, #7]           ; encoding: [0xab,0x73,0x40,0xf8]
; CHECK: ldur	w0, [x0, #2]            ; encoding: [0x00,0x20,0x40,0xb8]
; CHECK: ldur	w0, [x0, #-256]         ; encoding: [0x00,0x00,0x50,0xb8]
; CHECK: ldur	b2, [x1, #-2]           ; encoding: [0x22,0xe0,0x5f,0x3c]
; CHECK: ldur	h3, [x2, #3]            ; encoding: [0x43,0x30,0x40,0x7c]
; CHECK: ldur	h3, [x3, #-4]           ; encoding: [0x63,0xc0,0x5f,0x7c]
; CHECK: ldur	s3, [x4, #3]            ; encoding: [0x83,0x30,0x40,0xbc]
; CHECK: ldur	s3, [x5, #-4]           ; encoding: [0xa3,0xc0,0x5f,0xbc]
; CHECK: ldur	d4, [x6, #4]            ; encoding: [0xc4,0x40,0x40,0xfc]
; CHECK: ldur	d4, [x7, #-8]           ; encoding: [0xe4,0x80,0x5f,0xfc]
; CHECK: ldur	q5, [x8, #8]            ; encoding: [0x05,0x81,0xc0,0x3c]
; CHECK: ldur	q5, [x9, #-16]          ; encoding: [0x25,0x01,0xdf,0x3c]

  str x11, [x29, #-8]
  str x11, [x29, #7]
  str w0, [x0, #2]
  str w0, [x0, #-256]
  str b2, [x1, #-2]
  str h3, [x2, #3]
  str h3, [x3, #-4]
  str s3, [x4, #3]
  str s3, [x5, #-4]
  str d4, [x6, #4]
  str d4, [x7, #-8]
  str q5, [x8, #8]
  str q5, [x9, #-16]

; CHECK: stur	x11, [x29, #-8]          ; encoding: [0xab,0x83,0x1f,0xf8]
; CHECK: stur	x11, [x29, #7]           ; encoding: [0xab,0x73,0x00,0xf8]
; CHECK: stur	w0, [x0, #2]            ; encoding: [0x00,0x20,0x00,0xb8]
; CHECK: stur	w0, [x0, #-256]         ; encoding: [0x00,0x00,0x10,0xb8]
; CHECK: stur	b2, [x1, #-2]           ; encoding: [0x22,0xe0,0x1f,0x3c]
; CHECK: stur	h3, [x2, #3]            ; encoding: [0x43,0x30,0x00,0x7c]
; CHECK: stur	h3, [x3, #-4]           ; encoding: [0x63,0xc0,0x1f,0x7c]
; CHECK: stur	s3, [x4, #3]            ; encoding: [0x83,0x30,0x00,0xbc]
; CHECK: stur	s3, [x5, #-4]           ; encoding: [0xa3,0xc0,0x1f,0xbc]
; CHECK: stur	d4, [x6, #4]            ; encoding: [0xc4,0x40,0x00,0xfc]
; CHECK: stur	d4, [x7, #-8]           ; encoding: [0xe4,0x80,0x1f,0xfc]
; CHECK: stur	q5, [x8, #8]            ; encoding: [0x05,0x81,0x80,0x3c]
; CHECK: stur	q5, [x9, #-16]          ; encoding: [0x25,0x01,0x9f,0x3c]

  ldrb w3, [x1, #-1]
  ldrh w4, [x2, #1]
  ldrh w5, [x3, #-1]
  ldrsb w6, [x4, #-1]
  ldrsb x7, [x5, #-1]
  ldrsh w8, [x6, #1]
  ldrsh w9, [x7, #-1]
  ldrsh x1, [x8, #1]
  ldrsh x2, [x9, #-1]
  ldrsw x3, [x10, #10]
  ldrsw x4, [x11, #-1]

; CHECK: ldurb	w3, [x1, #-1]           ; encoding: [0x23,0xf0,0x5f,0x38]
; CHECK: ldurh	w4, [x2, #1]            ; encoding: [0x44,0x10,0x40,0x78]
; CHECK: ldurh	w5, [x3, #-1]           ; encoding: [0x65,0xf0,0x5f,0x78]
; CHECK: ldursb	w6, [x4, #-1]           ; encoding: [0x86,0xf0,0xdf,0x38]
; CHECK: ldursb	x7, [x5, #-1]           ; encoding: [0xa7,0xf0,0x9f,0x38]
; CHECK: ldursh	w8, [x6, #1]            ; encoding: [0xc8,0x10,0xc0,0x78]
; CHECK: ldursh	w9, [x7, #-1]           ; encoding: [0xe9,0xf0,0xdf,0x78]
; CHECK: ldursh	x1, [x8, #1]            ; encoding: [0x01,0x11,0x80,0x78]
; CHECK: ldursh	x2, [x9, #-1]           ; encoding: [0x22,0xf1,0x9f,0x78]
; CHECK: ldursw	x3, [x10, #10]          ; encoding: [0x43,0xa1,0x80,0xb8]
; CHECK: ldursw	x4, [x11, #-1]          ; encoding: [0x64,0xf1,0x9f,0xb8]

  strb w3, [x1, #-1]
  strh w4, [x2, #1]
  strh w5, [x3, #-1]

; CHECK: sturb	w3, [x1, #-1]           ; encoding: [0x23,0xf0,0x1f,0x38]
; CHECK: sturh	w4, [x2, #1]            ; encoding: [0x44,0x10,0x00,0x78]
; CHECK: sturh	w5, [x3, #-1]           ; encoding: [0x65,0xf0,0x1f,0x78]
