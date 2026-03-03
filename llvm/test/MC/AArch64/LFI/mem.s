// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-no-guard-elim %s | FileCheck %s

ldr x0, [sp]
// CHECK: ldr x0, [sp]

ldr x0, [sp, #8]
// CHECK: ldr x0, [sp, #8]

ldp x0, x1, [sp, #8]
// CHECK: ldp x0, x1, [sp, #8]

str x0, [sp]
// CHECK: str x0, [sp]

str x0, [sp, #8]
// CHECK: str x0, [sp, #8]

stp x0, x1, [sp, #8]
// CHECK: stp x0, x1, [sp, #8]

ldur x0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldur x0, [x28]

stur x0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stur x0, [x28]

ldp x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28]

stp x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp x0, x1, [x28]

ldr x0, [x1]
// CHECK: ldr x0, [x27, w1, uxtw]

ldr x0, [x1, #8]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldr x0, [x28, #8]

ldr x0, [x1, #8]!
// CHECK:      add x1, x1, #8
// CHECK-NEXT: ldr x0, [x27, w1, uxtw]

str x0, [x1, #8]!
// CHECK:      add x1, x1, #8
// CHECK-NEXT: str x0, [x27, w1, uxtw]

ldr x0, [x1, #-8]!
// CHECK:      sub x1, x1, #8
// CHECK-NEXT: ldr x0, [x27, w1, uxtw]

str x0, [x1, #-8]!
// CHECK:      sub x1, x1, #8
// CHECK-NEXT: str x0, [x27, w1, uxtw]

ldr x0, [x1], #8
// CHECK:      ldr x0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #8

str x0, [x1], #8
// CHECK:      str x0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #8

ldr x0, [x1], #-8
// CHECK:      ldr x0, [x27, w1, uxtw]
// CHECK-NEXT: sub x1, x1, #8

str x0, [x1], #-8
// CHECK:      str x0, [x27, w1, uxtw]
// CHECK-NEXT: sub x1, x1, #8

ldr x0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: ldr x0, [x27, w26, uxtw]

ldr x0, [x1, x2, lsl #3]
// CHECK:      add x26, x1, x2, lsl #3
// CHECK-NEXT: ldr x0, [x27, w26, uxtw]

ldr x0, [x1, x2, sxtx #0]
// CHECK:      add x26, x1, x2, sxtx
// CHECK-NEXT: ldr x0, [x27, w26, uxtw]

ldr x0, [x1, x2, sxtx #3]
// CHECK:      add x26, x1, x2, sxtx #3
// CHECK-NEXT: ldr x0, [x27, w26, uxtw]

ldr x0, [x1, w2, uxtw]
// CHECK:      add x26, x1, w2, uxtw
// CHECK-NEXT: ldr x0, [x27, w26, uxtw]

ldr x0, [x1, w2, uxtw #3]
// CHECK:      add x26, x1, w2, uxtw #3
// CHECK-NEXT: ldr x0, [x27, w26, uxtw]

ldr x0, [x1, w2, sxtw]
// CHECK:      add x26, x1, w2, sxtw
// CHECK-NEXT: ldr x0, [x27, w26, uxtw]

ldr x0, [x1, w2, sxtw #3]
// CHECK:      add x26, x1, w2, sxtw #3
// CHECK-NEXT: ldr x0, [x27, w26, uxtw]

ldp x0, x1, [sp], #8
// CHECK: ldp x0, x1, [sp], #8

ldp x0, x1, [x2], #8
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28]
// CHECK-NEXT: add x2, x2, #8

ldp x0, x1, [x2, #8]!
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28, #8]
// CHECK-NEXT: add x2, x2, #8

ldp x0, x1, [x2], #-8
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28]
// CHECK-NEXT: sub x2, x2, #8

ldp x0, x1, [x2, #-8]!
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28, #-8]
// CHECK-NEXT: sub x2, x2, #8

stp x0, x1, [x2, #-8]!
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp x0, x1, [x28, #-8]
// CHECK-NEXT: sub x2, x2, #8

ld3 { v0.4s, v1.4s, v2.4s }, [x0], #48
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.4s, v1.4s, v2.4s }, [x28]
// CHECK-NEXT: add x0, x0, #48

st2 { v1.8b, v2.8b }, [x14], #16
// CHECK:      add x28, x27, w14, uxtw
// CHECK-NEXT: st2 { v1.8b, v2.8b }, [x28]
// CHECK-NEXT: add x14, x14, #16

st2 { v1.8b, v2.8b }, [x14]
// CHECK:      add x28, x27, w14, uxtw
// CHECK-NEXT: st2 { v1.8b, v2.8b }, [x28]

ld1 { v0.s }[1], [x8]
// CHECK:      add x28, x27, w8, uxtw
// CHECK-NEXT: ld1 { v0.s }[1], [x28]

ld1r { v3.2d }, [x9]
// CHECK:      add x28, x27, w9, uxtw
// CHECK-NEXT: ld1r { v3.2d }, [x28]

ld1 { v0.s }[1], [x8], x10
// CHECK:      add x28, x27, w8, uxtw
// CHECK-NEXT: ld1 { v0.s }[1], [x28]
// CHECK-NEXT: add x8, x8, x10

ldaxr x0, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldaxr x0, [x28]

stlxr w15, w17, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stlxr w15, w17, [x28]

ldr w4, [sp, w3, uxtw #2]
// CHECK:      add x26, sp, w3, uxtw #2
// CHECK-NEXT: ldr w4, [x27, w26, uxtw]

stxrb w11, w10, [x8]
// CHECK:      add x28, x27, w8, uxtw
// CHECK-NEXT: stxrb w11, w10, [x28]

ldr x0, [x0, :got_lo12:x]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ldr x0, [x28, :got_lo12:x]

prfm pstl1strm, [x10]
// CHECK: prfm pstl1strm, [x27, w10, uxtw]

prfm pstl1strm, [x10, x11]
// CHECK:      add x26, x10, x11
// CHECK-NEXT: prfm pstl1strm, [x27, w26, uxtw]

// Byte loads/stores
ldrb w0, [x1]
// CHECK: ldrb w0, [x27, w1, uxtw]

ldrb w0, [x1, #1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldrb w0, [x28, #1]

strb w0, [x1]
// CHECK: strb w0, [x27, w1, uxtw]

ldrsb w0, [x1]
// CHECK: ldrsb w0, [x27, w1, uxtw]

ldrsb x0, [x1]
// CHECK: ldrsb x0, [x27, w1, uxtw]

// Halfword loads/stores
ldrh w0, [x1]
// CHECK: ldrh w0, [x27, w1, uxtw]

ldrh w0, [x1, #2]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldrh w0, [x28, #2]

strh w0, [x1]
// CHECK: strh w0, [x27, w1, uxtw]

ldrsh w0, [x1]
// CHECK: ldrsh w0, [x27, w1, uxtw]

ldrsh x0, [x1]
// CHECK: ldrsh x0, [x27, w1, uxtw]

// Word loads/stores
ldr w0, [x1]
// CHECK: ldr w0, [x27, w1, uxtw]

ldr w0, [x1, #4]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldr w0, [x28, #4]

str w0, [x1]
// CHECK: str w0, [x27, w1, uxtw]

ldrsw x0, [x1]
// CHECK: ldrsw x0, [x27, w1, uxtw]

// 32-bit pairs
ldp w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp w0, w1, [x28]

stp w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp w0, w1, [x28]

// Unscaled loads/stores
ldurb w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldurb w0, [x28]

ldurb w0, [x1, #1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldurb w0, [x28, #1]

ldursb w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldursb w0, [x28]

ldurh w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldurh w0, [x28]

ldursh w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldursh w0, [x28]

ldur w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldur w0, [x28]

ldursw x0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldursw x0, [x28]

sturb w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: sturb w0, [x28]

sturh w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: sturh w0, [x28]

stur w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stur w0, [x28]

// Byte pre/post-index
ldrb w0, [x1, #1]!
// CHECK:      add x1, x1, #1
// CHECK-NEXT: ldrb w0, [x27, w1, uxtw]

ldrb w0, [x1], #1
// CHECK:      ldrb w0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #1

strb w0, [x1, #1]!
// CHECK:      add x1, x1, #1
// CHECK-NEXT: strb w0, [x27, w1, uxtw]

strb w0, [x1], #1
// CHECK:      strb w0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #1

// Halfword pre/post-index
ldrh w0, [x1, #2]!
// CHECK:      add x1, x1, #2
// CHECK-NEXT: ldrh w0, [x27, w1, uxtw]

ldrh w0, [x1], #2
// CHECK:      ldrh w0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #2

// Word pre/post-index
ldr w0, [x1, #4]!
// CHECK:      add x1, x1, #4
// CHECK-NEXT: ldr w0, [x27, w1, uxtw]

ldr w0, [x1], #4
// CHECK:      ldr w0, [x27, w1, uxtw]
// CHECK-NEXT: add x1, x1, #4

// Register offset with different sizes
ldrb w0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: ldrb w0, [x27, w26, uxtw]

ldrh w0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: ldrh w0, [x27, w26, uxtw]

ldrh w0, [x1, x2, lsl #1]
// CHECK:      add x26, x1, x2, lsl #1
// CHECK-NEXT: ldrh w0, [x27, w26, uxtw]

ldr w0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: ldr w0, [x27, w26, uxtw]

ldr w0, [x1, x2, lsl #2]
// CHECK:      add x26, x1, x2, lsl #2
// CHECK-NEXT: ldr w0, [x27, w26, uxtw]

strb w0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: strb w0, [x27, w26, uxtw]

strh w0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: strh w0, [x27, w26, uxtw]

str w0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: str w0, [x27, w26, uxtw]

str x0, [x1, x2]
// CHECK:      add x26, x1, x2
// CHECK-NEXT: str x0, [x27, w26, uxtw]

// Sign/zero extension variants
ldrb w0, [x1, w2, uxtw]
// CHECK:      add x26, x1, w2, uxtw
// CHECK-NEXT: ldrb w0, [x27, w26, uxtw]

ldrb w0, [x1, w2, sxtw]
// CHECK:      add x26, x1, w2, sxtw
// CHECK-NEXT: ldrb w0, [x27, w26, uxtw]

ldrh w0, [x1, w2, uxtw]
// CHECK:      add x26, x1, w2, uxtw
// CHECK-NEXT: ldrh w0, [x27, w26, uxtw]

ldrh w0, [x1, w2, uxtw #1]
// CHECK:      add x26, x1, w2, uxtw #1
// CHECK-NEXT: ldrh w0, [x27, w26, uxtw]

ldr w0, [x1, w2, sxtw #2]
// CHECK:      add x26, x1, w2, sxtw #2
// CHECK-NEXT: ldr w0, [x27, w26, uxtw]

// Byte loads with #0 shift (shift amount omitted in output).
ldrsb x0, [x1, x2, sxtx #0]
// CHECK:      add x26, x1, x2, sxtx{{$}}
// CHECK-NEXT: ldrsb x0, [x27, w26, uxtw]

ldrsb w0, [x1, x2, sxtx #0]
// CHECK:      add x26, x1, x2, sxtx{{$}}
// CHECK-NEXT: ldrsb w0, [x27, w26, uxtw]

ldrsb w0, [x1, w2, sxtw #0]
// CHECK:      add x26, x1, w2, sxtw{{$}}
// CHECK-NEXT: ldrsb w0, [x27, w26, uxtw]

ldrsb x0, [x1, w2, uxtw #0]
// CHECK:      add x26, x1, w2, uxtw{{$}}
// CHECK-NEXT: ldrsb x0, [x27, w26, uxtw]

ldrsh x0, [x1, x2, sxtx #1]
// CHECK:      add x26, x1, x2, sxtx #1
// CHECK-NEXT: ldrsh x0, [x27, w26, uxtw]

ldrsh w0, [x1, x2, sxtx #1]
// CHECK:      add x26, x1, x2, sxtx #1
// CHECK-NEXT: ldrsh w0, [x27, w26, uxtw]

ldrsh w0, [x1, w2, sxtw #1]
// CHECK:      add x26, x1, w2, sxtw #1
// CHECK-NEXT: ldrsh w0, [x27, w26, uxtw]

ldrsw x0, [x1, x2, sxtx #2]
// CHECK:      add x26, x1, x2, sxtx #2
// CHECK-NEXT: ldrsw x0, [x27, w26, uxtw]

ldrsw x0, [x1, w2, sxtw #2]
// CHECK:      add x26, x1, w2, sxtw #2
// CHECK-NEXT: ldrsw x0, [x27, w26, uxtw]

// 32-bit pair pre/post-index
ldp w0, w1, [x2], #8
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp w0, w1, [x28]
// CHECK-NEXT: add x2, x2, #8

ldp w0, w1, [x2, #8]!
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp w0, w1, [x28, #8]
// CHECK-NEXT: add x2, x2, #8

stp w0, w1, [x2], #8
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp w0, w1, [x28]
// CHECK-NEXT: add x2, x2, #8

stp w0, w1, [x2, #8]!
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp w0, w1, [x28, #8]
// CHECK-NEXT: add x2, x2, #8
