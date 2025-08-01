// RUN: not llvm-mc -triple x86_64-apple-macosx -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

// CHECK: error: cannot encode offset of section

        .data
        .long 1
        .zero 0x100000000
        .const
        .long 1
