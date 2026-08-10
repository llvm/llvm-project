# RUN: not llvm-mc %s -triple=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1 -check-prefix=ASM

        .text
        .option pic2
        .set noreorder
        .set mips16
        .cpload $25
# ASM: :[[@LINE-1]]:17: error: .cpload is not supported in Mips16 mode

        .set nomips16
        .set reorder
        .cpload $25
# ASM: :[[@LINE-1]]:9: warning: .cpload should be inside a noreorder section

        .set noreorder
        .cpload $32
# ASM: :[[@LINE-1]]:17: error: invalid register

        .cpload $foo
# ASM: :[[@LINE-1]]:17: error: expected register containing function address

        .cpload bar
# ASM: :[[@LINE-1]]:17: error: expected register containing function address

        .cpload $25 foobar
# ASM: :[[@LINE-1]]:21: error: unexpected token, expected end of statement
