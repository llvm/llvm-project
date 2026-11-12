# RUN: not llvm-mc %s -triple mips64-unknown-unknown 2>%t1
# RUN:   FileCheck %s < %t1 -check-prefix=ASM

        .text
        .option pic2
t1:
        .cpsetup $bar, 8, __cerror
# ASM: :[[@LINE-1]]:18: error: expected register containing function address
        .cpsetup $33, 8, __cerror
# ASM: :[[@LINE-1]]:18: error: invalid register
        .cpsetup $31, foo, __cerror
# ASM: :[[@LINE-1]]:23: error: expected save register or stack offset
        .cpsetup $31, $32, __cerror
# ASM: :[[@LINE-1]]:23: error: invalid register
        .cpsetup $25, $2, 4
# ASM: :[[@LINE-1]]:28: error: expected symbol
        .cpsetup $25, $2, 4+65
# ASM: :[[@LINE-1]]:31: error: expected symbol
        .cpsetup $25, $2, foo+4
# ASM: :[[@LINE-1]]:32: error: expected symbol
