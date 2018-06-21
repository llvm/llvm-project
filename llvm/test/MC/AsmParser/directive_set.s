# RUN: llvm-mc -triple i386-unknown-elf %s | FileCheck %s

# CHECK: TEST0:
# CHECK: a = 0
# CHECK-NOT: .no_dead_strip a
TEST0:  
        .set a, 0
        
# CHECK: TEST1:
# CHECK: a = 0
# CHECK-NOT: .no_dead_strip a
TEST1:  
        .equ a, 0

