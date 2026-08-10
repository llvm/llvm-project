// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r - | FileCheck %s

a:
    .section foo
    c = b
b:
    // CHECK: 0x0 R_X86_64_PC32 .text 0x0
    .long a - c
