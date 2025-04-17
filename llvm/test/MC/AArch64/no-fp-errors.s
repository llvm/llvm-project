// RUN: not llvm-mc -triple aarch64-none-eabi -mattr=-fp-armv8 < %s 2>&1 | FileCheck %s --implicit-check-not error

  ldr      s0, [x0]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str      q0, [x0]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  fmov     d0, xzr
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  ldnp     s0, s1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldnp     d0, d1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldnp     q0, q1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  ldp       s0, s1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldp       d0, d1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldp       q0, q1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  ldp    s0, s1, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldp    d0, d1, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldp    q0, q1, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  ldp       s0, s1, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldp       d0, d1, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldp       q0, q1, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8


  ldr    b0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr    h0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr    s0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr    d0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr    q0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  ldr     b0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     h0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     s0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     d0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     q0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  ldr     b0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     h0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     s0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     d0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     q0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  ldr     b0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     h0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     s0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     d0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr     q0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  ldr      b0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr      h0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr      s0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr      d0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr      q0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

label:
  ldr       s0, label
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr       d0, label
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  ldr       q0, label
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  stnp     s0, s1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  stnp     d0, d1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  stnp     q0, q1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  stp       s0, s1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  stp       d0, d1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  stp       q0, q1, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  stp    s0, s1, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  stp    d0, d1, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  stp    q0, q1, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  stp     s0, s1, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  stp     d0, d1, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  stp     q0, q1, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  str    b0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str    h0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str    s0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str    d0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str    q0, [x0], #16
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  str     b0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     h0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     s0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     d0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     q0, [x0, #16]!
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  str     b0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     h0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     s0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     d0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     q0, [x0, #16]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  str     b0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     h0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     s0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     d0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     q0, [x0, x1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  str     b0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     h0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     s0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     d0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str     q0, [x0, w1, sxtw]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  mrs x0, FPCR
// CHECK: [[@LINE-1]]:11: error: expected readable system register
  mrs x0, FPSR
// CHECK: [[@LINE-1]]:11: error: expected readable system register
  msr FPCR, x0
// CHECK: [[@LINE-1]]:7: error: expected writable system register or pstate
  msr FPSR, x0
// CHECK: [[@LINE-1]]:7: error: expected writable system register or pstate

  ldr s0, [x0, #1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  str q0, [x0, #1]
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  fmov s0, #0.0
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  fmov d0, #0.0
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  mvn v0.8b, v1.8b
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mvn v0.16b, v1.16b
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  mov v0.16b, v1.16b
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mov v0.8h, v1.8h
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mov v0.4s, v1.4s
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mov v0.2d, v1.2d
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  mov v0.8b, v1.8b
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mov v0.4h, v1.4h
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mov v0.2s, v1.2s
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mov v0.1d, v1.1d
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  faclt v0.4h, v1.4h, v2.4h
// CHECK: [[@LINE-1]]:3: error: instruction requires: fullfp16 neon
  faclt v0.8h, v1.8h, v2.8h
// CHECK: [[@LINE-1]]:3: error: instruction requires: fullfp16 neon
  faclt v0.2s, v1.2s, v2.2s
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  faclt v0.4s, v1.4s, v2.4s
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  faclt v0.2d, v1.2d, v2.2d
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  cmls d0, d1, d2
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  cmle d0, d1, d2
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  cmlo d0, d1, d2
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  cmlt d0, d1, d2
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  fcmle s0, s1, s2
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  fcmle d0, d1, d2
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  fcmlt s0, s1, s2
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  fcmlt d0, d1, d2
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  facle s0, s1, s2
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  facle d0, d1, d2
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  faclt s0, s1, s2
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8
  faclt d0, d1, d2
// CHECK: [[@LINE-1]]:3: error: instruction requires: fp-armv8

  bic v0.4h, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  bic v0.8h, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  bic v0.2s, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  bic v0.4s, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  bic.4h v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  bic.8h v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  bic.2s v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  bic.4s v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  orr v0.4h, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  orr v0.8h, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  orr v0.2s, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  orr v0.4s, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  orr.4h v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  orr.8h v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  orr.2s v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  orr.4s v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  movi v0.4h, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  movi v0.8h, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  movi v0.2s, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  movi v0.4s, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  movi.4h v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  movi.8h v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  movi.2s v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  movi.4s v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  mvni v0.4h, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mvni v0.8h, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mvni v0.2s, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mvni v0.4s, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  mvni.4h v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mvni.8h v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mvni.2s v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  mvni.4s v0, #42
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  sxtl.8h v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  sxtl.4s v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  sxtl.2d v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  sxtl2.8h v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  sxtl2.4s v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  sxtl2.2d v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  uxtl.8h v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  uxtl.4s v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  uxtl.2d v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon

  uxtl2.8h v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  uxtl2.4s v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
  uxtl2.2d v0, v1
// CHECK: [[@LINE-1]]:3: error: instruction requires: neon
