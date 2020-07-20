// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r - | FileCheck %s

        .global zed
foo:
        nop
bar:
        nop
zed:
        mov zed+(bar-foo), %eax

// CHECK:       0x5 R_X86_64_32S zed 0x1
