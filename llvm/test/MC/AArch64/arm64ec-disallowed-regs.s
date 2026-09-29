// RUN: llvm-mc -triple arm64ec-pc-windows-msvc < %s 2> %t.log
// RUN: FileCheck %s --check-prefix=CHECK-ERR < %t.log
// RUN: llvm-mc -triple aarch64-windows-msvc < %s > %t.log 2>&1 
// RUN: FileCheck %s --check-prefix=CHECK-NOEC < %t.log

// ---- disallowed x registers ----
orr x13, x0, x1             // x13
// CHECK-ERR: warning: register X13 is disallowed on ARM64EC.
orr x14, x2, x3             // x14
// CHECK-ERR: warning: register X14 is disallowed on ARM64EC.
orr x4, x23, x5             // x23
// CHECK-ERR: warning: register X23 is disallowed on ARM64EC.
orr x6, x7, x24             // x24
// CHECK-ERR: warning: register X24 is disallowed on ARM64EC.
orr x28, x8, x9             // x28
// CHECK-ERR: warning: register X28 is disallowed on ARM64EC.

// ---- disallowed w registers ----
orr w0, w13, w1             // w13
// CHECK-ERR: warning: register W13 is disallowed on ARM64EC.
orr w14, w2, w3             // w14
// CHECK-ERR: warning: register W14 is disallowed on ARM64EC.
orr w4, w23, w5             // w23
// CHECK-ERR: warning: register W23 is disallowed on ARM64EC.
orr w6, w7, w24             // W24
// CHECK-ERR: warning: register W24 is disallowed on ARM64EC.
orr w28, w8, w9             // w28
// CHECK-ERR: warning: register W28 is disallowed on ARM64EC.

// ---- disallowed vector registers ----
orn v1.8b, v16.8b, v2.8b        // v16
// CHECK-ERR: warning: register D16 is disallowed on ARM64EC.
orn v2.16b, v17.16b, v3.16b     // v17
// CHECK-ERR: warning: register Q17 is disallowed on ARM64EC.
orn v3.8b, v18.8b, v4.8b        // v18
// CHECK-ERR: warning: register D18 is disallowed on ARM64EC.
orn v4.16b, v19.16b, v5.16b     // v19
// CHECK-ERR: warning: register Q19 is disallowed on ARM64EC.
orn v5.8b, v20.8b, v6.8b        // v20
// CHECK-ERR: warning: register D20 is disallowed on ARM64EC.
orn v21.8b, v6.8b, v7.8b        // v21
// CHECK-ERR: warning: register D21 is disallowed on ARM64EC.
orn v7.16b, v8.16b, v22.16b     // v22
// CHECK-ERR: warning: register Q22 is disallowed on ARM64EC.
orn v23.8b, v8.8b, v9.8b        // v23
// CHECK-ERR: warning: register D23 is disallowed on ARM64EC.
orn v9.16b, v24.16b, v10.16b    // v24
// CHECK-ERR: warning: register Q24 is disallowed on ARM64EC.
orn v10.8b, v25.8b, v11.8b      // v25
// CHECK-ERR: warning: register D25 is disallowed on ARM64EC.
orn v11.8b, v12.8b, v26.8b      // v26
// CHECK-ERR: warning: register D26 is disallowed on ARM64EC.
orn v12.8b, v27.8b, v13.8b      // v27
// CHECK-ERR: warning: register D27 is disallowed on ARM64EC.
orn v13.16b, v28.16b, v14.16b   // v28
// CHECK-ERR: warning: register Q28 is disallowed on ARM64EC.
orn v14.8b, v29.8b, v15.8b      // v29
// CHECK-ERR: warning: register D29 is disallowed on ARM64EC.
orn v15.8b, v30.8b, v15.8b      // v30
// CHECK-ERR: warning: register D30 is disallowed on ARM64EC.
orn v1.16b, v31.16b, v1.16b     // v31
// CHECK-ERR: warning: register Q31 is disallowed on ARM64EC.

// ---- random tests on h, b, d, s registers ----
orn.16b v1, v16, v2
// CHECK-ERR: warning: register Q16 is disallowed on ARM64EC.
str d17, [x0]
// CHECK-ERR: warning: register D17 is disallowed on ARM64EC.
fmul d2, d18, d11
// CHECK-ERR: warning: register D18 is disallowed on ARM64EC.
clz.8h v3, v19
// CHECK-ERR: warning: register Q19 is disallowed on ARM64EC.
add.4s v0, v20, v1
// CHECK-ERR: warning: register Q20 is disallowed on ARM64EC.
add.2d v0, v20, v1
// CHECK-ERR: warning: register Q20 is disallowed on ARM64EC.
str b17, [x28]
// CHECK-ERR: warning: register B17 is disallowed on ARM64EC.
// CHECK-ERR: warning: register X28 is disallowed on ARM64EC.
addv h21, v22.4h
// CHECK-ERR: warning: register H21 is disallowed on ARM64EC.
// CHECK-ERR: warning: register D22 is disallowed on ARM64EC.
mov w14, v24.s[0] 
// CHECK-ERR: warning: register W14 is disallowed on ARM64EC.
// CHECK-ERR: warning: register Q24 is disallowed on ARM64EC.
add x13, x14, x28
// CHECK-ERR: warning: register X13 is disallowed on ARM64EC.
// CHECK-ERR: warning: register X14 is disallowed on ARM64EC.
// CHECK-ERR: warning: register X28 is disallowed on ARM64EC.

// CHECK-NOEC-NOT: warning:
