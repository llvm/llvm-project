// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

adrp x0, :tlsdesc:var
ldr x1, [x0, :tlsdesc_lo12:var]
add x0, x0, :tlsdesc_lo12:var
.tlsdesccall var
blr x1
// CHECK:      add x0, x0, :tlsdesc_lo12:var
// CHECK-NOT:  .tlsdesccall
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: .tlsdesccall var
// CHECK-NEXT: blr x28
