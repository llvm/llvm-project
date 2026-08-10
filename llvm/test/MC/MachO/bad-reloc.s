// RUN: not llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - 2>&1 | FileCheck %s

.quad (0 - undef)

// CHECK: error: unsupported relocation expression
