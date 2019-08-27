// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null \
// RUN: 2>&1 | FileCheck %s

// CHECK: [[@LINE+2]]:{{[0-9]+}}: error: Cannot represent a difference across sections
// CHECK-NEXT: call foo - bar
        call foo - bar

        .section .foo
foo:
        .section .bar
bar:
