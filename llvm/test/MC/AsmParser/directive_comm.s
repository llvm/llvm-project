# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .comm a,6,2
# CHECK: .comm b,8
# CHECK: .comm c,8
TEST0:  
        .comm a, 4+2, 2
        .comm b,8
        .common c,8
