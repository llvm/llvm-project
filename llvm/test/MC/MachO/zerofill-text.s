// RUN: not llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o /dev/null 2>&1 | FileCheck %s

        .text
        .zerofill __TEXT, __const, zfill, 2, 1

// CHECK: 4:27: error: The usage of .zerofill is restricted to sections of ZEROFILL type. Use .zero or .space instead.
// CHECK-NEXT: .zerofill __TEXT, __const, zfill, 2, 1
