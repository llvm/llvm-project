# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: .byte 65
        .ascii "A"
# CHECK: .byte 66
        .ASCII "B"
# CHECK: .byte 67
        .aScIi "C"

# Note: using 2byte because it is an alias
# CHECK: .short 4660
        .2byte 0x1234
# CHECK: .short 4661
        .2BYTE 0x1235
# CHECK: .short 4662
        .2bYtE 0x1236
