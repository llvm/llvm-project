# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .data
# CHECK: TEST0:
TEST0:  
        .ascii

# CHECK: TEST1:
TEST1:  
        .asciz

# CHECK: TEST2:
# CHECK: .byte 65
TEST2:  
        .ascii "A"

# CHECK: TEST3:
# CHECK: .byte 66
# CHECK: .byte 0
# CHECK: .byte 67
# CHECK: .byte 0
TEST3:  
        .asciz "B", "C"
        
# CHECK: TEST4:
# CHECK: .asciz "\001\001\007\0008\001\0001\200"
TEST4:  
        .ascii "\1\01\07\08\001\0001\200\0"
        
# CHECK: TEST5:
# CHECK: .ascii "\b\f\n\r\t\\\""
TEST5:
        .ascii "\b\f\n\r\t\\\""
        
# CHECK: TEST6:
# CHECK: .byte 66
# CHECK: .byte 0
# CHECK: .byte 67
# CHECK: .byte 0
TEST6:
        .string "B", "C"
