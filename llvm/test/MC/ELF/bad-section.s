// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null 2>%t
// RUN: FileCheck --input-file=%t %s

// CHECK: error: unexpected token in directive
// CHECK: .section "foo"-bar

// test that we don't accept this, as gas doesn't.

.section "foo"-bar
