// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null 2>%t
// RUN: FileCheck --input-file=%t %s

// CHECK: :[[#@LINE+5]]:15: error: expected end of directive
// CHECK: .section "foo"-bar

// test that we don't accept this, as gas doesn't.

.section "foo"-bar
