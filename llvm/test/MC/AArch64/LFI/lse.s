// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-no-guard-elim %s | FileCheck %s
// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-no-guard-elim -mattr=+no-lfi-loads %s | FileCheck %s --check-prefix=NOLOADS

.arch_extension lse

// LDADD variants
// Atomics are both loads and stores, so +no-lfi-loads must still sandbox them.
ldadd x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldadd x0, x1, [x28]
// NOLOADS:      add x28, x27, w2, uxtw
// NOLOADS-NEXT: ldadd x0, x1, [x28]

ldadd w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldadd w0, w1, [x28]

ldadda x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldadda x0, x1, [x28]

ldaddal x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldaddal x0, x1, [x28]

ldaddl x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldaddl x0, x1, [x28]

ldaddab w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldaddab w0, w1, [x28]

ldaddah w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldaddah w0, w1, [x28]

// LDCLR variants
ldclr x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldclr x0, x1, [x28]

ldclra x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldclra x0, x1, [x28]

ldclral x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldclral x0, x1, [x28]

ldclrl x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldclrl x0, x1, [x28]

// LDEOR variants
ldeor x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldeor x0, x1, [x28]

ldeora x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldeora x0, x1, [x28]

// LDSET variants
ldset x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldset x0, x1, [x28]

ldseta x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldseta x0, x1, [x28]

// SWP variants
swp x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: swp x0, x1, [x28]
// NOLOADS:      swp x0, x1, [x28]

swpa x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: swpa x0, x1, [x28]

swpal x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: swpal x0, x1, [x28]

swpl x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: swpl x0, x1, [x28]

swpab w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: swpab w0, w1, [x28]

swpah w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: swpah w0, w1, [x28]

swpalb w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: swpalb w0, w1, [x28]

swpalh w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: swpalh w0, w1, [x28]

swpal w0, w0, [x1]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: swpal w0, w0, [x28]

// CAS variants
cas x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: cas x0, x1, [x28]
// NOLOADS:      cas x0, x1, [x28]

casa x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: casa x0, x1, [x28]

casal x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: casal x0, x1, [x28]

casl x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: casl x0, x1, [x28]

casab w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: casab w0, w1, [x28]

casah w0, w1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: casah w0, w1, [x28]

// CASP variants (pair)
casp x0, x1, x2, x3, [x4]
// CHECK:      add x28, x27, w4, uxtw
// CHECK-NEXT: casp x0, x1, x2, x3, [x28]

caspa x0, x1, x2, x3, [x4]
// CHECK:      add x28, x27, w4, uxtw
// CHECK-NEXT: caspa x0, x1, x2, x3, [x28]

caspal x0, x1, x2, x3, [x4]
// CHECK:      add x28, x27, w4, uxtw
// CHECK-NEXT: caspal x0, x1, x2, x3, [x28]

caspl x0, x1, x2, x3, [x4]
// CHECK:      add x28, x27, w4, uxtw
// CHECK-NEXT: caspl x0, x1, x2, x3, [x28]

caspal w0, w1, w2, w3, [x4]
// CHECK:      add x28, x27, w4, uxtw
// CHECK-NEXT: caspal w0, w1, w2, w3, [x28]

// SP-relative atomics (no sandboxing needed)
ldadd x0, x1, [sp]
// CHECK: ldadd x0, x1, [sp]

swp x0, x1, [sp]
// CHECK: swp x0, x1, [sp]

cas x0, x1, [sp]
// CHECK: cas x0, x1, [sp]
