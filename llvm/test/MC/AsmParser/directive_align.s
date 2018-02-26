# RUN: not llvm-mc -triple i386-apple-darwin9 %s 2> %t.err | FileCheck %s
# RUN: FileCheck < %t.err %s --check-prefix=CHECK-WARN

# CHECK: TEST0:
# CHECK: .p2align 1
TEST0:  
        .align 1

# CHECK: TEST1:
# CHECK: .p2alignl 3, 0x0, 2
TEST1:  
        .align32 3,,2

# CHECK: TEST2:
# CHECK: .balign 3, 10
TEST2:  
        .balign 3,10

# CHECK-WARN: p2align directive with no operand(s) is ignored
TEST3:  
        .p2align
