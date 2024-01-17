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
