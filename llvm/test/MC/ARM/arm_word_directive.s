@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown %s | FileCheck %s

@ CHECK: TEST0:
@ CHECK: .long 3
TEST0:  
        .word 3
