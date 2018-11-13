# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s | llvm-objdump -t - | FileCheck %s

# CHECK: 0000000000000004 g     O *COM*  00000004 C
        .comm   C,4,4
        .comm   C,4,4
