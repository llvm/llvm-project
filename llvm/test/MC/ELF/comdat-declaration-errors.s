// RUN: not llvm-mc -triple x86_64-pc-linux-gnu %s \
// RUN:   -filetype=obj -o %t.o 2>&1 | FileCheck %s

// Check we error out on incorrect COMDATs declarations
// and not just silently ingnore them.

// CHECK:      error: invalid group name
// CHECK-NEXT: .section .foo,"G",@progbits,-abc,comdat

// CHECK:      error: invalid linkage
// CHECK-NEXT: .section .bar,"G",@progbits,abc,-comdat

.section .foo,"G",@progbits,-abc,comdat
.section .bar,"G",@progbits,abc,-comdat
