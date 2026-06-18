// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-no-guard-elim %s | FileCheck %s

// FP/SIMD scalar loads (zero offset -> RoW)
ldr b0, [x1]
// CHECK: ldr b0, [x27, w1, uxtw]

ldr h0, [x1]
// CHECK: ldr h0, [x27, w1, uxtw]

ldr s0, [x1]
// CHECK: ldr s0, [x27, w1, uxtw]

ldr d0, [x1]
// CHECK: ldr d0, [x27, w1, uxtw]

ldr q0, [x1]
// CHECK: ldr q0, [x27, w1, uxtw]

// FP/SIMD scalar stores (zero offset -> RoW)
str b0, [x1]
// CHECK: str b0, [x27, w1, uxtw]

str h0, [x1]
// CHECK: str h0, [x27, w1, uxtw]

str s0, [x1]
// CHECK: str s0, [x27, w1, uxtw]

str d0, [x1]
// CHECK: str d0, [x27, w1, uxtw]

str q0, [x1]
// CHECK: str q0, [x27, w1, uxtw]

// FP loads with non-zero offset (demoted)
ldr s0, [x1, #4]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldr s0, [x28, #4]

ldr d0, [x1, #8]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldr d0, [x28, #8]

ldr q0, [x1, #16]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldr q0, [x28, #16]

// FP stores with non-zero offset (demoted)
str s0, [x1, #4]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: str s0, [x28, #4]

str d0, [x1, #8]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: str d0, [x28, #8]

str q0, [x1, #16]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: str q0, [x28, #16]

// FP pre-index
ldr s0, [x1, #4]!
// CHECK:      add x1, x1, #4
// CHECK-NEXT: ldr s0, [x27, w1, uxtw]

ldr d0, [x1, #8]!
// CHECK:      add x1, x1, #8
// CHECK-NEXT: ldr d0, [x27, w1, uxtw]

ldr q0, [x1, #16]!
// CHECK:      add x1, x1, #16
// CHECK-NEXT: ldr q0, [x27, w1, uxtw]

str s0, [x1, #4]!
// CHECK:      add x1, x1, #4
// CHECK-NEXT: str s0, [x27, w1, uxtw]

// FP post-index
ldr s0, [x1], #4
// CHECK:      ldr s0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #4

ldr d0, [x1], #8
// CHECK:      ldr d0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #8

ldr q0, [x1], #16
// CHECK:      ldr q0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #16

str d0, [x1], #8
// CHECK:      str d0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #8

// FP register offset
ldr s0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: ldr s0, [x27, w26, uxtw]

ldr d0, [x1, x2, lsl #3]
// CHECK:      add x26, x1, x2, lsl #3
// CHECK-NEXT: ldr d0, [x27, w26, uxtw]

ldr q0, [x1, x2, lsl #4]
// CHECK:      add x26, x1, x2, lsl #4
// CHECK-NEXT: ldr q0, [x27, w26, uxtw]

str s0, [x1, x2, lsl #2]
// CHECK:      add x26, x1, x2, lsl #2
// CHECK-NEXT: str s0, [x27, w26, uxtw]

str d0, [x1, w2, sxtw]
// CHECK:      add x26, x1, w2, sxtw
// CHECK-NEXT: str d0, [x27, w26, uxtw]

str q0, [x1, w2, uxtw #4]
// CHECK:      add x26, x1, w2, uxtw #4
// CHECK-NEXT: str q0, [x27, w26, uxtw]

// FP unscaled offset
ldur s0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldur s0, [x28]

ldur d0, [x1, #8]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldur d0, [x28, #8]

ldur q0, [x1, #-16]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldur q0, [x28, #-16]

stur s0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stur s0, [x28]

stur d0, [x1, #8]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stur d0, [x28, #8]

// FP pair loads/stores
ldp s0, s1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp s0, s1, [x28]

ldp d0, d1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp d0, d1, [x28]

ldp q0, q1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp q0, q1, [x28]

stp s0, s1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp s0, s1, [x28]

stp d0, d1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp d0, d1, [x28]

stp q0, q1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp q0, q1, [x28]

// FP pair with offset
ldp s0, s1, [x2, #8]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp s0, s1, [x28, #8]

ldp d0, d1, [x2, #16]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp d0, d1, [x28, #16]

// FP pair pre/post-index
ldp s0, s1, [x2], #8
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp s0, s1, [x28]
// CHECK-NEXT: add x2, x2, #8

ldp d0, d1, [x2, #16]!
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp d0, d1, [x28, #16]
// CHECK-NEXT: add x2, x2, #16

stp q0, q1, [x2], #32
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp q0, q1, [x28]
// CHECK-NEXT: add x2, x2, #32

stp d0, d1, [x2, #-16]!
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp d0, d1, [x28, #-16]
// CHECK-NEXT: sub x2, x2, #16

// SP-relative FP loads (no sandboxing needed)
ldr s0, [sp]
// CHECK: ldr s0, [sp]

ldr d0, [sp, #8]
// CHECK: ldr d0, [sp, #8]

ldp q0, q1, [sp, #32]
// CHECK: ldp q0, q1, [sp, #32]
