; RUN: llvm-mc -triple arm64-apple-darwin -show-encoding < %s | FileCheck %s

foo:

;-----------------------------------------------------------------------------
; Unconditional branch (register) instructions.
;-----------------------------------------------------------------------------

  ret
; CHECK: encoding: [0xc0,0x03,0x5f,0xd6]
  ret x1
; CHECK: encoding: [0x20,0x00,0x5f,0xd6]
  drps
; CHECK: encoding: [0xe0,0x03,0xbf,0xd6]
  eret
; CHECK: encoding: [0xe0,0x03,0x9f,0xd6]
  br  x5
; CHECK: encoding: [0xa0,0x00,0x1f,0xd6]
  blr x9
; CHECK: encoding: [0x20,0x01,0x3f,0xd6]
  bl  L1
; CHECK: bl L1   ; encoding: [A,A,A,0b100101AA]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_call26

;-----------------------------------------------------------------------------
; Contitional branch instructions.
;-----------------------------------------------------------------------------

  b     L1
; CHECK: b L1      ; encoding: [A,A,A,0b000101AA]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch26
  b.eq  L1
; CHECK: b.eq L1   ; encoding: [0bAAA00000,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.ne  L1
; CHECK: b.ne L1   ; encoding: [0bAAA00001,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.cs  L1
; CHECK: b.hs L1   ; encoding: [0bAAA00010,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.cc  L1
; CHECK: b.lo L1   ; encoding: [0bAAA00011,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.mi  L1
; CHECK: b.mi L1   ; encoding: [0bAAA00100,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.pl  L1
; CHECK: b.pl L1   ; encoding: [0bAAA00101,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.vs  L1
; CHECK: b.vs L1   ; encoding: [0bAAA00110,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.vc  L1
; CHECK: b.vc L1   ; encoding: [0bAAA00111,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.hi  L1
; CHECK: b.hi L1   ; encoding: [0bAAA01000,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.ls  L1
; CHECK: b.ls L1   ; encoding: [0bAAA01001,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.ge  L1
; CHECK: b.ge L1   ; encoding: [0bAAA01010,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.lt  L1
; CHECK: b.lt L1   ; encoding: [0bAAA01011,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.gt  L1
; CHECK: b.gt L1   ; encoding: [0bAAA01100,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.le  L1
; CHECK: b.le L1   ; encoding: [0bAAA01101,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
  b.al  L1
; CHECK: b.al L1      ; encoding: [0bAAA01110,A,A,0x54]
; CHECK: fixup A - offset: 0, value: L1, kind: fixup_aarch64_pcrel_branch19
L1:
  b #28
; CHECK: b #28
  b.lt #28
; CHECK: b.lt #28
  b.cc #1048572
; CHECK: b.lo	#1048572                ; encoding: [0xe3,0xff,0x7f,0x54]
  b #134217724
; CHECK: b	#134217724              ; encoding: [0xff,0xff,0xff,0x15]
  b #-134217728
; CHECK: b	#-134217728             ; encoding: [0x00,0x00,0x00,0x16]

;-----------------------------------------------------------------------------
; Compare-and-branch instructions.
;-----------------------------------------------------------------------------

  cbz w1, foo
; CHECK: encoding: [0bAAA00001,A,A,0x34]
  cbz x1, foo
; CHECK: encoding: [0bAAA00001,A,A,0xb4]
  cbnz w2, foo
; CHECK: encoding: [0bAAA00010,A,A,0x35]
  cbnz x2, foo
; CHECK: encoding: [0bAAA00010,A,A,0xb5]
  cbz w1, #28
; CHECK: cbz w1, #28
  cbz     w20, #1048572
; CHECK: cbz	w20, #1048572           ; encoding: [0xf4,0xff,0x7f,0x34]
  cbnz x2, #-1048576
; CHECK: cbnz	x2, #-1048576           ; encoding: [0x02,0x00,0x80,0xb5]


;-----------------------------------------------------------------------------
; Bit-test-and-branch instructions.
;-----------------------------------------------------------------------------

  tbz x1, #3, foo
; CHECK: encoding: [0bAAA00001,A,0b00011AAA,0x36]
  tbnz x1, #63, foo
; CHECK: encoding: [0bAAA00001,A,0b11111AAA,0xb7]

  tbz w1, #3, foo
; CHECK: encoding: [0bAAA00001,A,0b00011AAA,0x36]
  tbnz w1, #31, foo
; CHECK: encoding: [0bAAA00001,A,0b11111AAA,0x37]

  tbz w1, #3, #28
; CHECK: tbz w1, #3, #28
  tbz w3, #5, #32764
; CHECK: tbz	w3, #5, #32764          ; encoding: [0xe3,0xff,0x2b,0x36]
  tbnz x3, #8, #-32768
; CHECK: tbnz	w3, #8, #-32768         ; encoding: [0x03,0x00,0x44,0x37]

;-----------------------------------------------------------------------------
; Exception generation instructions.
;-----------------------------------------------------------------------------

  brk   #1
; CHECK: encoding: [0x20,0x00,0x20,0xd4]
  dcps1 #2
; CHECK: encoding: [0x41,0x00,0xa0,0xd4]
  dcps2 #3
; CHECK: encoding: [0x62,0x00,0xa0,0xd4]
  hlt   #5
; CHECK: encoding: [0xa0,0x00,0x40,0xd4]
  hvc   #6
; CHECK: encoding: [0xc2,0x00,0x00,0xd4]
  svc   #8
; CHECK: encoding: [0x01,0x01,0x00,0xd4]

; The immediate defaults to zero for DCPSn
  dcps1
  dcps2

; CHECK: dcps1                     ; encoding: [0x01,0x00,0xa0,0xd4]
; CHECK: dcps2                     ; encoding: [0x02,0x00,0xa0,0xd4]

