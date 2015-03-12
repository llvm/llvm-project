# RUN: not llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 2>%t1
# RUN:   FileCheck %s < %t1

        .text
        .set pop
# CHECK: :[[@LINE-1]]:14: error: .set pop with no .set push
        .set push
        .set pop
        .set pop
# CHECK: :[[@LINE-1]]:14: error: .set pop with no .set push
        .set push foo
# CHECK: :[[@LINE-1]]:19: error: unexpected token, expected end of statement
        .set pop bar
# CHECK: :[[@LINE-1]]:18: error: unexpected token, expected end of statement
