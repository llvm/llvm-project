// RUN: llvm-mc -triple arm64ec-pc-windows-msvc < %s 2> %t.log
// RUN: FileCheck %s --check-prefix=CHECK-ERR < %t.log


// ---- disallowed x registers ----
orr x13, x0, x1             // x13
// CHECK-ERR: warning: this instruction uses disallowed registers.
orr x14, x2, x3             // x14
// CHECK-ERR: warning: this instruction uses disallowed registers.
orr x4, x23, x5             // x23
// CHECK-ERR: warning: this instruction uses disallowed registers.
orr x6, x7, x24             // x24
// CHECK-ERR: warning: this instruction uses disallowed registers.
orr x28, x8, x9             // x28
// CHECK-ERR: warning: this instruction uses disallowed registers.

// ---- disallowed w registers ----
orr w0, w13, w1             // w13
// CHECK-ERR: warning: this instruction uses disallowed registers.
orr w14, w2, w3             // w14
// CHECK-ERR: warning: this instruction uses disallowed registers.
orr w4, w23, w5             // w23
// CHECK-ERR: warning: this instruction uses disallowed registers.
orr w6, w7, w24             // w24
// CHECK-ERR: warning: this instruction uses disallowed registers.
orr w28, w8, w9             // w28
// CHECK-ERR: warning: this instruction uses disallowed registers.

// ---- disallowed vector registers ----
orn v1.8b, v16.8b, v2.8b        // v16
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v2.16b, v17.16b, v3.16b     // v17
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v3.8b, v18.8b, v4.8b        // v18
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v4.16b, v19.16b, v5.16b     // v19
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v5.8b, v20.8b, v6.8b        // v20
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v21.8b, v6.8b, v7.8b        // v21
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v7.16b, v8.16b, v22.16b     // v22
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v23.8b, v8.8b, v9.8b        // v23
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v9.16b, v24.16b, v10.16b    // v24
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v10.8b, v25.8b, v11.8b      // v25
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v11.8b, v12.8b, v26.8b      // v26
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v12.8b, v27.8b, v13.8b      // v27
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v13.16b, v28.16b, v14.16b   // v28
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v14.8b, v29.8b, v15.8b      // v29
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v15.8b, v30.8b, v15.8b      // v30
// CHECK-ERR: warning: this instruction uses disallowed registers.
orn v1.16b, v31.16b, v1.16b     // v31
// CHECK-ERR: warning: this instruction uses disallowed registers.

// ---- random tests on h, b, d, s registers ----
orn.16b v1, v16, v2
// CHECK-ERR: warning: this instruction uses disallowed registers.
mov.4h v17, v8
// CHECK-ERR: warning: this instruction uses disallowed registers.
fmul.2s v2, v18, v11
// CHECK-ERR: warning: this instruction uses disallowed registers.
clz.8h v3, v19
// CHECK-ERR: warning: this instruction uses disallowed registers.
add.4s v0, v20, v1
// CHECK-ERR: warning: this instruction uses disallowed registers.
add.2d v0, v20, v1
// CHECK-ERR: warning: this instruction uses disallowed registers.