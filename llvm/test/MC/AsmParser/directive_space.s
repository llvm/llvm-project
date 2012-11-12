# RUN: llvm-mc -triple i386-apple-darwin %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .space 1
TEST0:  
        .space 1

# CHECK: TEST1:
# CHECK: .space	2,3
TEST1:  
        .space 2, 3

# CHECK: TEST2:
# CHECK: .space 1
TEST2:
        .skip 1
