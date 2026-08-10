# RUN: not llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 2>%t1
# RUN:   FileCheck %s < %t1

# Check only one error per statement.

        .text
        .set pop
# CHECK: :[[@LINE-1]]:14: error: .set pop with no .set push
# CHECK-NOT: :[[@LINE-1]]:
        .set push
        .set pop
        .set pop
# CHECK: :[[@LINE-1]]:14: error: .set pop with no .set push
# CHECK-NOT: :[[@LINE-1]]:
        .set push foo
# CHECK: :[[@LINE-1]]:19: error: unexpected token, expected end of statement
# CHECK-NOT: :[[@LINE-1]]:
        .set pop bar
# CHECK: :[[@LINE-1]]:18: error: unexpected token, expected end of statement
# CHECK-NOT: :[[@LINE-1]]:
        .set hardfloat
        .set push
        .set softfloat
        add.s $f2, $f2, $f2
# CHECK: :[[@LINE-1]]:9: error: instruction requires a CPU feature not currently enabled
# CHECK-NOT: :[[@LINE-1]]:
        .set pop
        add.s $f2, $f2, $f2
# CHECK-NOT: :[[@LINE-1]]:9: error: instruction requires a CPU feature not currently enabled
# CHECK-NOT: :[[@LINE-1]]:
