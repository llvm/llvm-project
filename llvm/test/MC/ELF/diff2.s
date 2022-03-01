// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s 2> %t
// RUN: FileCheck -input-file %t %s

.global zed
        .data
foo:
        .text
        nop
bar:
        nop
zed:
// CHECK: expected relocatable expression
        mov zed+(bar-foo), %eax
