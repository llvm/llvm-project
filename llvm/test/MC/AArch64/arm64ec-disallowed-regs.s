// RUN: llvm-mc -triple arm64ec-pc-windows-msvc < %s 2> %t.log
// RUN: FileCheck %s --check-prefix=CHECK-ERR < %t.log
// RUN: llvm-mc -triple aarch64-windows-msvc < %s > %t.log 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-NOEC --implicit-check-not=warning: < %t.log

// ---- disallowed x registers ----
orr x13, x0, x1             // x13
// CHECK-ERR: warning: register x13 is disallowed on ARM64EC.
orr x14, x2, x3             // x14
// CHECK-ERR: warning: register x14 is disallowed on ARM64EC.
orr x4, x23, x5             // x23
// CHECK-ERR: warning: register x23 is disallowed on ARM64EC.
orr x6, x7, x24             // x24
// CHECK-ERR: warning: register x24 is disallowed on ARM64EC.
orr x28, x8, x9             // x28
// CHECK-ERR: warning: register x28 is disallowed on ARM64EC.

// ---- disallowed w registers ----
orr w0, w13, w1             // w13
// CHECK-ERR: warning: register w13 is disallowed on ARM64EC.
orr w14, w2, w3             // w14
// CHECK-ERR: warning: register w14 is disallowed on ARM64EC.
orr w4, w23, w5             // w23
// CHECK-ERR: warning: register w23 is disallowed on ARM64EC.
orr w6, w7, w24             // w24
// CHECK-ERR: warning: register w24 is disallowed on ARM64EC.
orr w28, w8, w9             // w28
// CHECK-ERR: warning: register w28 is disallowed on ARM64EC.

// ---- disallowed vector registers ----
orn v1.8b, v16.8b, v2.8b        // v16
// CHECK-ERR: warning: register v16 is disallowed on ARM64EC.
orn v2.16b, v17.16b, v3.16b     // v17
// CHECK-ERR: warning: register v17 is disallowed on ARM64EC.
orn v3.8b, v18.8b, v4.8b        // v18
// CHECK-ERR: warning: register v18 is disallowed on ARM64EC.
orn v4.16b, v19.16b, v5.16b     // v19
// CHECK-ERR: warning: register v19 is disallowed on ARM64EC.
orn v5.8b, v20.8b, v6.8b        // v20
// CHECK-ERR: warning: register v20 is disallowed on ARM64EC.
orn v21.8b, v6.8b, v7.8b        // v21
// CHECK-ERR: warning: register v21 is disallowed on ARM64EC.
orn v7.16b, v8.16b, v22.16b     // v22
// CHECK-ERR: warning: register v22 is disallowed on ARM64EC.
orn v23.8b, v8.8b, v9.8b        // v23
// CHECK-ERR: warning: register v23 is disallowed on ARM64EC.
orn v9.16b, v24.16b, v10.16b    // v24
// CHECK-ERR: warning: register v24 is disallowed on ARM64EC.
orn v10.8b, v25.8b, v11.8b      // v25
// CHECK-ERR: warning: register v25 is disallowed on ARM64EC.
orn v11.8b, v12.8b, v26.8b      // v26
// CHECK-ERR: warning: register v26 is disallowed on ARM64EC.
orn v12.8b, v27.8b, v13.8b      // v27
// CHECK-ERR: warning: register v27 is disallowed on ARM64EC.
orn v13.16b, v28.16b, v14.16b   // v28
// CHECK-ERR: warning: register v28 is disallowed on ARM64EC.
orn v14.8b, v29.8b, v15.8b      // v29
// CHECK-ERR: warning: register v29 is disallowed on ARM64EC.
orn v15.8b, v30.8b, v15.8b      // v30
// CHECK-ERR: warning: register v30 is disallowed on ARM64EC.
orn v1.16b, v31.16b, v1.16b     // v31
// CHECK-ERR: warning: register v31 is disallowed on ARM64EC.

// ---- random tests on h, b, d, s registers ----
orn.16b v1, v16, v2
// CHECK-ERR: warning: register v16 is disallowed on ARM64EC.
mov.4h v17, v8
// CHECK-ERR: warning: register v17 is disallowed on ARM64EC.
fmul.2s v2, v18, v11
// CHECK-ERR: warning: register v18 is disallowed on ARM64EC.
clz.8h v3, v19
// CHECK-ERR: warning: register v19 is disallowed on ARM64EC.
add.4s v0, v20, v1
// CHECK-ERR: warning: register v20 is disallowed on ARM64EC.
add.2d v0, v20, v1
// CHECK-ERR: warning: register v20 is disallowed on ARM64EC.

// CHECK-NOEC: .text
// CHECK-NOEC: orr x13, x0, x1
// CHECK-NOEC: orr x14, x2, x3
// CHECK-NOEC: orr x4, x23, x5
// CHECK-NOEC: orr x6, x7, x24
// CHECK-NOEC: orr x28, x8, x9 

// CHECK-NOEC: orr w0, w13, w1
// CHECK-NOEC: orr w14, w2, w3
// CHECK-NOEC: orr w4, w23, w5
// CHECK-NOEC: orr w6, w7, w24
// CHECK-NOEC: orr w28, w8, w9

// CHECK-NOEC: orn v1.8b, v16.8b, v2.8b
// CHECK-NOEC: orn v2.16b, v17.16b, v3.16b
// CHECK-NOEC: orn v3.8b, v18.8b, v4.8b
// CHECK-NOEC: orn v4.16b, v19.16b, v5.16b
// CHECK-NOEC: orn v5.8b, v20.8b, v6.8b
// CHECK-NOEC: orn v21.8b, v6.8b, v7.8b
// CHECK-NOEC: orn v7.16b, v8.16b, v22.16b
// CHECK-NOEC: orn v23.8b, v8.8b, v9.8b
// CHECK-NOEC: orn v9.16b, v24.16b, v10.16b
// CHECK-NOEC: orn v10.8b, v25.8b, v11.8b
// CHECK-NOEC: orn v11.8b, v12.8b, v26.8b
// CHECK-NOEC: orn v12.8b, v27.8b, v13.8b
// CHECK-NOEC: orn v13.16b, v28.16b, v14.16b
// CHECK-NOEC: orn v14.8b, v29.8b, v15.8b
// CHECK-NOEC: orn v15.8b, v30.8b, v15.8b
// CHECK-NOEC: orn v1.16b, v31.16b, v1.16b

// CHECK-NOEC: orn v1.16b, v16.16b, v2.16b
// CHECK-NOEC: mov v17.8b, v8.8b
// CHECK-NOEC: fmul v2.2s, v18.2s, v11.2s
// CHECK-NOEC: clz v3.8h, v19.8h
// CHECK-NOEC: add v0.4s, v20.4s, v1.4s
// CHECK-NOEC: add v0.2d, v20.2d, v1.2d
