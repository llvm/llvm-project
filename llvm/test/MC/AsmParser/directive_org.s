# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .org 1, 0
TEST0:  
        .org 1

# CHECK: TEST1:
# CHECK: .org 1, 3
TEST1:  
        .org 1, 3
