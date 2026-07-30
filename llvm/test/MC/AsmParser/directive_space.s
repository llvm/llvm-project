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
# CHECK: .space	1,3
TEST2:  
        .space -(2 > 0), 3

# CHECK: TEST3:
# CHECK: .space 1
TEST3:
        .skip 1

# CHECK: TEST4
# CHECK: .space TEST0-TEST1
TEST4:
        .skip TEST0 - TEST1

# CHECK: TEST5
# CHECK: .space -((TEST0-TEST1)>0)
TEST5:
        .skip -((TEST0 - TEST1) > 0)
