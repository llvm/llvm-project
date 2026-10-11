// RUN: llvm-mc -triple aarch64_lfi -filetype=obj %s -o /dev/null
// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

// CHECK:      mov	x30, x0
// CHECK:      add	x30, x27, w30, uxtw
// CHECK:      next_func:
// CHECK-NEXT: nop

.file 1 "debug-info.s"
mov x30, x0
.loc 1 10 0
next_func:
  nop
