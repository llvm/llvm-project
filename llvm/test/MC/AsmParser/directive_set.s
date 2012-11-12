# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: a = 0
TEST0:  
        .set a, 0
        
# CHECK: TEST1:
# CHECK: a = 0
TEST1:  
        .equ a, 0

