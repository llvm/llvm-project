// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t
// RUN: FileCheck < %t %s

.endr

// CHECK: unmatched '.endr' directive
