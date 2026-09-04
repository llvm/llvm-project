// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

// Ensure hint instruction is emitted immediately before store

stshh keep
stlr w1, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: stshh keep
// CHECK-NEXT: stlr w1, [x28]

stshh strm
strb w2, [x0, w1, sxtw]
// CHECK:      add x26, x0, w1, sxtw
// CHECK-NEXT: stshh strm
// CHECK-NEXT: strb w2, [x27, w26, uxtw]

stshh keep
stur w0, [x1, #-256]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stshh keep
// CHECK-NEXT: stur w0, [x28, #-256]

stshh strm
strh w2, [x0, x1, lsl #1]
// CHECK:      add x26, x0, x1, lsl #1
// CHECK-NEXT: stshh strm
// CHECK-NEXT: strh w2, [x27, w26, uxtw]

stshh keep
label_after_hint:
stlr w1, [x0]
// CHECK:      label_after_hint:
// CHECK-NEXT: add x28, x27, w0, uxtw
// CHECK-NEXT: stshh keep
// CHECK-NEXT: stlr w1, [x28]

// The hint should always appear before a store instruction, but
// test that we do not change the order if this was not the case.

stshh keep
b label_branch
label_branch:
stlr w1, [x0]
// CHECK:      stshh keep
// CHECK-NEXT: b label_branch
// CHECK-NEXT: label_branch:
// CHECK-NEXT: add x28, x27, w0, uxtw
// CHECK-NEXT: stlr w1, [x28]

stshh keep
ldp x0, x1, [x2]
// CHECK:      stshh keep
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28]

stshh strm
add x0, x0, #1
stlr w1, [x0]
// CHECK:      stshh strm
// CHECK-NEXT: add x0, x0, #1
// CHECK-NEXT: add x28, x27, w0, uxtw
// CHECK-NEXT: stlr w1, [x28]
