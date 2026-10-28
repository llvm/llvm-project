// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

svc #0
// CHECK:      mov x26, x30
// CHECK-NEXT: ldur x30, [x27, #-8]
// CHECK-NEXT: blr x30
// CHECK-NEXT: add x30, x27, w26, uxtw

dc zva, x0
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: dc zva, x28

dc zva, x5
// CHECK:      add x28, x27, w5, uxtw
// CHECK-NEXT: dc zva, x28

dc cvac, x0
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: dc cvac, x28

dc cvau, x1
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: dc cvau, x28

dc civac, x2
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: dc civac, x28

ic ivau, x4
// CHECK:      add x28, x27, w4, uxtw
// CHECK-NEXT: ic ivau, x28
