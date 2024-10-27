# RUN: not llvm-mc -triple i386-apple-darwin9 %s 2> %t.err | FileCheck %s
# RUN: FileCheck < %t.err %s --check-prefix=CHECK-WARN

        .data

# CHECK: TEST0:
# CHECK: .p2align 1
TEST0:  
        .align 1

# CHECK: TEST1:
# CHECK: .p2alignl 3, 0x0, 2
TEST1:  
        .align32 3,,2

# CHECK: TEST2:
# CHECK-WARN: error: alignment must be a power of 2
# CHECK: .p2align 1, 0xa
TEST2:  
        .balign 3,10

# CHECK-WARN: p2align directive with no operand(s) is ignored
TEST3:  
        .p2align

# CHECK: TEST4:
# CHECK: .p2align  31, 0x90
# CHECK-WARN: error: alignment must be smaller than 2**32
TEST4:  
        .balign 0x100000000, 0x90

# CHECK: TEST5:
# CHECK: .p2align  31, 0x90
# CHECK-WARN: error: alignment must be a power of 2
# CHECK-WARN: error: alignment must be smaller than 2**32
TEST5:  
        .balign 0x100000001, 0x90

