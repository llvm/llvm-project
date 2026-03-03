// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

foo:
  b foo
// CHECK: b foo

  br x0
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: br x28

  blr x0
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: blr x28

  ret
// CHECK: ret

  ret x0
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ret x28
