# RUN: llvm-mc -triple i386-apple-darwin9 %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .align 1
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
