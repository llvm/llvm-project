; RUN: not llvm-mc -triple arm64-apple-darwin -show-encoding < %s 2> %t | FileCheck %s
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

foo:

; The first should encode as an expression. The second should error expecting
; a register.
  ldr x3, (foo + 4)
  ldr x3, [foo + 4]
; CHECK:  ldr x3, foo+4               ; encoding: [0bAAA00011,A,A,0x58]
; CHECK:                              ;   fixup A - offset: 0, value: foo+4, kind: fixup_aarch64_ldr_pcrel_imm19
; CHECK-ERRORS: error: invalid operand for instruction

; The last argument should be flagged as an error.  rdar://9576009
  ld4.8b	{v0, v1, v2, v3}, [x0], #33
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: ld4.8b	{v0, v1, v2, v3}, [x0], #33


        ldr x0, [x0, #804]
        ldr w0, [x0, #802]
        ldr x0, [x0, #804]!
        ldr w0, [w0, #301]!
        ldr x0, [x0], #804
        ldr w0, [w0], #301

        ldp w3, w4, [x5, #11]!
        ldp x3, x4, [x5, #12]!
        ldp q3, q4, [x5, #12]!
        ldp w3, w4, [x5], #11
        ldp x3, x4, [x5], #12
        ldp q3, q4, [x5], #12

        ldur x0, [x1, #-257]

; CHECK-ERRORS: error: index must be an integer in range [-256, 255].
; CHECK-ERRORS:         ldr x0, [x0, #804]
; CHECK-ERRORS:                 ^
; CHECK-ERRORS: error: index must be an integer in range [-256, 255].
; CHECK-ERRORS:         ldr w0, [x0, #802]
; CHECK-ERRORS:                 ^
; CHECK-ERRORS: error: index must be an integer in range [-256, 255].
; CHECK-ERRORS:         ldr x0, [x0, #804]!
; CHECK-ERRORS:                 ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS:         ldr w0, [w0, #301]!
; CHECK-ERRORS:                  ^
; CHECK-ERRORS: error: index must be an integer in range [-256, 255].
; CHECK-ERRORS:         ldr x0, [x0], #804
; CHECK-ERRORS:                       ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS:         ldr w0, [w0], #301
; CHECK-ERRORS:                  ^
; CHECK-ERRORS: error: index must be a multiple of 4 in range [-256, 252].
; CHECK-ERRORS:         ldp w3, w4, [x5, #11]!
; CHECK-ERRORS:                     ^
; CHECK-ERRORS: error: index must be a multiple of 8 in range [-512, 504].
; CHECK-ERRORS:         ldp x3, x4, [x5, #12]!
; CHECK-ERRORS:                     ^
; CHECK-ERRORS: error: index must be a multiple of 16 in range [-1024, 1008].
; CHECK-ERRORS:         ldp q3, q4, [x5, #12]!
; CHECK-ERRORS:                     ^
; CHECK-ERRORS: error: index must be a multiple of 4 in range [-256, 252].
; CHECK-ERRORS:         ldp w3, w4, [x5], #11
; CHECK-ERRORS:                           ^
; CHECK-ERRORS: error: index must be a multiple of 8 in range [-512, 504].
; CHECK-ERRORS:         ldp x3, x4, [x5], #12
; CHECK-ERRORS:                           ^
; CHECK-ERRORS: error: index must be a multiple of 16 in range [-1024, 1008].
; CHECK-ERRORS:         ldp q3, q4, [x5], #12
; CHECK-ERRORS:                           ^
; CHECK-ERRORS: error: index must be an integer in range [-256, 255].
; CHECK-ERRORS:         ldur x0, [x1, #-257]
; CHECK-ERRORS:                   ^


ldrb   w1, [x3, w3, sxtw #4]
ldrh   w1, [x3, w3, sxtw #4]
ldr    w1, [x3, w3, sxtw #4]
ldr    x1, [x3, w3, sxtw #4]
ldr    b1, [x3, w3, sxtw #4]
ldr    h1, [x3, w3, sxtw #4]
ldr    s1, [x3, w3, sxtw #4]
ldr    d1, [x3, w3, sxtw #4]
ldr    q1, [x3, w3, sxtw #1]

; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0
; CHECK-ERRORS:ldrb   w1, [x3, w3, sxtw #4]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #1
; CHECK-ERRORS:ldrh   w1, [x3, w3, sxtw #4]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #2
; CHECK-ERRORS:ldr    w1, [x3, w3, sxtw #4]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #3
; CHECK-ERRORS:ldr    x1, [x3, w3, sxtw #4]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0
; CHECK-ERRORS:ldr    b1, [x3, w3, sxtw #4]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #1
; CHECK-ERRORS:ldr    h1, [x3, w3, sxtw #4]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #2
; CHECK-ERRORS:ldr    s1, [x3, w3, sxtw #4]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #3
; CHECK-ERRORS:ldr    d1, [x3, w3, sxtw #4]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #4
; CHECK-ERRORS:ldr    q1, [x3, w3, sxtw #1]
; CHECK-ERRORS:           ^

; Check that register offset addressing modes only accept 32-bit offset
; registers when using uxtw/sxtw extends. Everything else requires a 64-bit
; register.
  str    d1, [x3, w3, sxtx #3]
  ldr    s1, [x3, d3, sxtx #2]

; CHECK-ERRORS: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #3
; CHECK-ERRORS:   str    d1, [x3, w3, sxtx #3]
; CHECK-ERRORS:                       ^
; CHECK-ERRORS: error: index must be an integer in range [-256, 255].
; CHECK-ERRORS:   ldr    s1, [x3, d3, sxtx #2]
; CHECK-ERRORS:                   ^

; Shift immediates range checking.
  sqrshrn b4, h9, #10
  rshrn v9.8b, v11.8h, #17
  sqrshrn v7.4h, v8.4s, #39
  uqshrn2 v4.4s, v5.2d, #67

; CHECK-ERRORS: error: immediate must be an integer in range [1, 8].
; CHECK-ERRORS:   sqrshrn b4, h9, #10
; CHECK-ERRORS:                   ^
; CHECK-ERRORS: error: immediate must be an integer in range [1, 8].
; CHECK-ERRORS:   rshrn v9.8b, v11.8h, #17
; CHECK-ERRORS:                        ^
; CHECK-ERRORS: error: immediate must be an integer in range [1, 16].
; CHECK-ERRORS:   sqrshrn v7.4h, v8.4s, #39
; CHECK-ERRORS:                         ^
; CHECK-ERRORS: error: immediate must be an integer in range [1, 32].
; CHECK-ERRORS:   uqshrn2 v4.4s, v5.2d, #67
; CHECK-ERRORS:                         ^


  st1.s4 {v14, v15}, [x2], #32
; CHECK-ERRORS: error: invalid type suffix for instruction
; CHECK-ERRORS: st1.s4 {v14, v15}, [x2], #32
; CHECK-ERRORS:     ^



; Load pair instructions where Rt==Rt2 and writeback load/store instructions
; where Rt==Rn or Rt2==Rn are unpredicatable.
  ldp x1, x2, [x2], #16
  ldp x2, x2, [x2], #16
  ldp w1, w2, [x2], #16
  ldp w2, w2, [x2], #16
  ldp x1, x1, [x2]
  ldp s1, s1, [x1], #8
  ldp s1, s1, [x1, #8]!
  ldp s1, s1, [x1, #8]
  ldp d1, d1, [x1], #16
  ldp d1, d1, [x1, #16]!
  ldp d1, d1, [x1, #16]
  ldp q1, q1, [x1], #32
  ldp q1, q1, [x1, #32]!
  ldp q1, q1, [x1, #32]

  ldr x2, [x2], #8
  ldr x2, [x2, #8]!
  ldr w2, [x2], #8
  ldr w2, [x2, #8]!

  str x2, [x2], #8
  str x2, [x2, #8]!
  str w2, [x2], #8
  str w2, [x2, #8]!

; CHECK-ERRORS: error: unpredictable LDP instruction, writeback base is also a destination
; CHECK-ERRORS:   ldp x1, x2, [x2], #16
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, writeback base is also a destination
; CHECK-ERRORS:   ldp x2, x2, [x2], #16
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDP instruction, writeback base is also a destination
; CHECK-ERRORS:   ldp w1, w2, [x2], #16
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, writeback base is also a destination
; CHECK-ERRORS:   ldp w2, w2, [x2], #16
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp x1, x1, [x2]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp s1, s1, [x1], #8
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp s1, s1, [x1, #8]!
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp s1, s1, [x1, #8]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp d1, d1, [x1], #16
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp d1, d1, [x1, #16]!
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp d1, d1, [x1, #16]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp q1, q1, [x1], #32
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp q1, q1, [x1, #32]!
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp q1, q1, [x1, #32]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDR instruction, writeback base is also a source
; CHECK-ERRORS:   ldr x2, [x2], #8
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDR instruction, writeback base is also a source
; CHECK-ERRORS:   ldr x2, [x2, #8]!
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDR instruction, writeback base is also a source
; CHECK-ERRORS:   ldr w2, [x2], #8
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDR instruction, writeback base is also a source
; CHECK-ERRORS:   ldr w2, [x2, #8]!
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable STR instruction, writeback base is also a source
; CHECK-ERRORS:   str x2, [x2], #8
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable STR instruction, writeback base is also a source
; CHECK-ERRORS:   str x2, [x2, #8]!
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable STR instruction, writeback base is also a source
; CHECK-ERRORS:   str w2, [x2], #8
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable STR instruction, writeback base is also a source
; CHECK-ERRORS:   str w2, [x2, #8]!
; CHECK-ERRORS:       ^

; The validity checking for shifted-immediate operands.  rdar://13174476
; Where the immediate is out of range.
  add w1, w2, w3, lsr #75

; CHECK-ERRORS: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
; CHECK-ERRORS: add w1, w2, w3, lsr #75
; CHECK-ERRORS:                      ^

; logical instructions on 32-bit regs with shift > 31 is not legal
orr w0, w0, w0, lsl #32
; CHECK-ERRORS: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
; CHECK-ERRORS:        orr w0, w0, w0, lsl #32
; CHECK-ERRORS:                        ^
eor w0, w0, w0, lsl #32
; CHECK-ERRORS: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
; CHECK-ERRORS:        eor w0, w0, w0, lsl #32
; CHECK-ERRORS:                        ^
and w0, w0, w0, lsl #32
; CHECK-ERRORS: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
; CHECK-ERRORS:        and w0, w0, w0, lsl #32
; CHECK-ERRORS:                        ^
ands w0, w0, w0, lsl #32
; CHECK-ERRORS: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
; CHECK-ERRORS:        ands w0, w0, w0, lsl #32
; CHECK-ERRORS:                        ^

; Relocated expressions should not be accepted for 32-bit adds or sub (imm)
add w3, w5, sym@PAGEOFF
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: add w3, w5, sym@PAGEOFF
; CHECK-ERRORS:             ^

adds w3, w5, sym@PAGEOFF
adds x9, x12, sym@PAGEOFF
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: adds w3, w5, sym@PAGEOFF
; CHECK-ERRORS:              ^
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: adds x9, x12, sym@PAGEOFF
; CHECK-ERRORS:               ^

sub x3, x5, sym@PAGEOFF
sub w20, w30, sym@PAGEOFF
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: sub x3, x5, sym@PAGEOFF
; CHECK-ERRORS:             ^
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: sub w20, w30, sym@PAGEOFF
; CHECK-ERRORS:               ^

subs w9, w10, sym@PAGEOFF
subs x20, x30, sym@PAGEOFF
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: subs w9, w10, sym@PAGEOFF
; CHECK-ERRORS:               ^
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: subs x20, x30, sym@PAGEOFF
; CHECK-ERRORS:                ^

tbl v0.8b, { v1 }, v0.8b
tbl v0.16b, { v1.8b, v2.8b, v3.8b }, v0.16b
tbx v3.16b, { v12.8b, v13.8b, v14.8b }, v6.8b
tbx v2.8b, { v0 }, v6.8b
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: tbl v0.8b, { v1 }, v0.8b
; CHECK-ERRORS:            ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: tbl v0.16b, { v1.8b, v2.8b, v3.8b }, v0.16b
; CHECK-ERRORS:             ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: tbx v3.16b, { v12.8b, v13.8b, v14.8b }, v6.8b
; CHECK-ERRORS:             ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: tbx v2.8b, { v0 }, v6.8b
; CHECK-ERRORS:            ^

b.c #0x4
; CHECK-ERRORS: error: invalid condition code
; CHECK-ERRORS: b.c #0x4
; CHECK-ERRORS:   ^

ic ialluis, x0
; CHECK-ERRORS: error: specified ic op does not use a register
ic iallu, x0
; CHECK-ERRORS: error: specified ic op does not use a register
ic ivau
; CHECK-ERRORS: error: specified ic op requires a register

dc zva
; CHECK-ERRORS: error: specified dc op requires a register
dc ivac
; CHECK-ERRORS: error: specified dc op requires a register
dc isw
; CHECK-ERRORS: error: specified dc op requires a register
dc cvac
; CHECK-ERRORS: error: specified dc op requires a register
dc csw
; CHECK-ERRORS: error: specified dc op requires a register
dc cvau
; CHECK-ERRORS: error: specified dc op requires a register
dc civac
; CHECK-ERRORS: error: specified dc op requires a register
dc cisw
; CHECK-ERRORS: error: specified dc op requires a register

at s1e1r
; CHECK-ERRORS: error: specified at op requires a register
at s1e2r
; CHECK-ERRORS: error: specified at op requires a register
at s1e3r
; CHECK-ERRORS: error: specified at op requires a register
at s1e1w
; CHECK-ERRORS: error: specified at op requires a register
at s1e2w
; CHECK-ERRORS: error: specified at op requires a register
at s1e3w
; CHECK-ERRORS: error: specified at op requires a register
at s1e0r
; CHECK-ERRORS: error: specified at op requires a register
at s1e0w
; CHECK-ERRORS: error: specified at op requires a register
at s12e1r
; CHECK-ERRORS: error: specified at op requires a register
at s12e1w
; CHECK-ERRORS: error: specified at op requires a register
at s12e0r
; CHECK-ERRORS: error: specified at op requires a register
at s12e0w
; CHECK-ERRORS: error: specified at op requires a register

tlbi vmalle1is, x0
; CHECK-ERRORS: error: specified tlbi op does not use a register
tlbi vmalle1, x0
; CHECK-ERRORS: error: specified tlbi op does not use a register
tlbi alle1is, x0
; CHECK-ERRORS: error: specified tlbi op does not use a register
tlbi alle2is, x0
; CHECK-ERRORS: error: specified tlbi op does not use a register
tlbi alle3is, x0
; CHECK-ERRORS: error: specified tlbi op does not use a register
tlbi alle1, x0
; CHECK-ERRORS: error: specified tlbi op does not use a register
tlbi alle2, x0
; CHECK-ERRORS: error: specified tlbi op does not use a register
tlbi alle3, x0
; CHECK-ERRORS: error: specified tlbi op does not use a register
tlbi vae1is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vae2is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vae3is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi aside1is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vaae1is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vale1is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vaale1is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vale2is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vale3is
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vae1
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vae2
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vae3
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi aside1
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vaae1
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vale1
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vale2
; CHECK-ERRORS: error: specified tlbi op requires a register
tlbi vale3
; CHECK-ERRORS: error: specified tlbi op requires a register


; Check that we give the proper "too few operands" diagnostic even when
; using short-form NEON.

  add.16b v0, v1, v2, v3
  add.8b v0, v1
  sub.8h v0, v1
  fadd.4s v0
  fmul.2s

; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS:   add.16b v0, v1, v2, v3
; CHECK-ERRORS:                       ^
; CHECK-ERRORS: error: too few operands for instruction
; CHECK-ERRORS:   add.8b v0, v1
; CHECK-ERRORS:   ^
; CHECK-ERRORS: error: too few operands for instruction
; CHECK-ERRORS:   sub.8h v0, v1
; CHECK-ERRORS:   ^
; CHECK-ERRORS: error: too few operands for instruction
; CHECK-ERRORS:   fadd.4s v0
; CHECK-ERRORS:   ^
; CHECK-ERRORS: error: too few operands for instruction
; CHECK-ERRORS:   fmul.2s
; CHECK-ERRORS:   ^

; Also for 2-operand instructions.

  frsqrte.4s v0, v1, v2
  frsqrte.2s v0
  frecpe.2d

; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS:   frsqrte.4s v0, v1, v2
; CHECK-ERRORS:                      ^
; CHECK-ERRORS: error: too few operands for instruction
; CHECK-ERRORS:   frsqrte.2s v0
; CHECK-ERRORS:   ^
; CHECK-ERRORS: error: too few operands for instruction
; CHECK-ERRORS:   frecpe.2d
; CHECK-ERRORS:   ^

; And check that we do the same for non-NEON instructions.

  b.ne
  b.eq 0, 0

; CHECK-ERRORS: error: too few operands for instruction
; CHECK-ERRORS:   b.ne
; CHECK-ERRORS:   ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS:   b.eq 0, 0
; CHECK-ERRORS:           ^

; Check that we give the proper "too few operands" diagnostic instead of
; asserting.

  ldr

; CHECK-ERRORS: error: too few operands for instruction
; CHECK-ERRORS:   ldr
; CHECK-ERRORS:   ^
