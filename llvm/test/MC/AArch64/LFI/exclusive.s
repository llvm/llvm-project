// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-no-guard-elim %s | FileCheck %s

// Load exclusive
ldxr x0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldxr x0, [x28]

ldxr w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldxr w0, [x28]

ldxrb w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldxrb w0, [x28]

ldxrh w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldxrh w0, [x28]

// Store exclusive
stxr w0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stxr w0, x1, [x28]

stxr w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stxr w0, w1, [x28]

stxrb w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stxrb w0, w1, [x28]

stxrh w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stxrh w0, w1, [x28]

// Load-acquire exclusive
ldaxr x0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldaxr x0, [x28]

ldaxr w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldaxr w0, [x28]

ldaxrb w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldaxrb w0, [x28]

ldaxrh w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldaxrh w0, [x28]

// Store-release exclusive
stlxr w0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stlxr w0, x1, [x28]

stlxr w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stlxr w0, w1, [x28]

stlxrb w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stlxrb w0, w1, [x28]

stlxrh w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stlxrh w0, w1, [x28]

// Exclusive pairs
ldxp x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldxp x0, x1, [x28]

ldxp w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldxp w0, w1, [x28]

stxp w0, x1, x2, [x3]
// CHECK:      add x28, x27, w3, uxtw
// CHECK-NEXT: stxp w0, x1, x2, [x28]

stxp w0, w1, w2, [x3]
// CHECK:      add x28, x27, w3, uxtw
// CHECK-NEXT: stxp w0, w1, w2, [x28]

ldaxp x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldaxp x0, x1, [x28]

stlxp w0, x1, x2, [x3]
// CHECK:      add x28, x27, w3, uxtw
// CHECK-NEXT: stlxp w0, x1, x2, [x28]

// Load-acquire / Store-release (non-exclusive)
ldar x0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldar x0, [x28]

ldar w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldar w0, [x28]

ldarb w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldarb w0, [x28]

ldarh w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldarh w0, [x28]

stlr x0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stlr x0, [x28]

stlr w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stlr w0, [x28]

stlrb w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stlrb w0, [x28]

stlrh w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: stlrh w0, [x28]

// SP-relative exclusive (no sandboxing needed)
ldxr x0, [sp]
// CHECK: ldxr x0, [sp]

stxr w0, x1, [sp]
// CHECK: stxr w0, x1, [sp]

ldar x0, [sp]
// CHECK: ldar x0, [sp]

stlr x0, [sp]
// CHECK: stlr x0, [sp]
