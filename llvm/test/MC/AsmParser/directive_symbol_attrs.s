# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .globl a
# CHECK: .globl b
TEST0:  
        .globl a, b
