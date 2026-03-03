// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

// Authenticated instructions are expanded to authenticate + guard + branch.

.arch_extension pauth

retaa
// CHECK:      autiasp
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

retab
// CHECK:      autibsp
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

braa x0, x1
// CHECK:      autia x0, x1
// CHECK-NEXT: add x28, x27, w0, uxtw
// CHECK-NEXT: br x28

braaz x2
// CHECK:      autiza x2
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: br x28

brab x3, x4
// CHECK:      autib x3, x4
// CHECK-NEXT: add x28, x27, w3, uxtw
// CHECK-NEXT: br x28

brabz x5
// CHECK:      autizb x5
// CHECK-NEXT: add x28, x27, w5, uxtw
// CHECK-NEXT: br x28

blraa x0, x1
// CHECK:      autia x0, x1
// CHECK-NEXT: add x28, x27, w0, uxtw
// CHECK-NEXT: blr x28

blraaz x2
// CHECK:      autiza x2
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: blr x28

blrab x3, x4
// CHECK:      autib x3, x4
// CHECK-NEXT: add x28, x27, w3, uxtw
// CHECK-NEXT: blr x28

blrabz x5
// CHECK:      autizb x5
// CHECK-NEXT: add x28, x27, w5, uxtw
// CHECK-NEXT: blr x28
