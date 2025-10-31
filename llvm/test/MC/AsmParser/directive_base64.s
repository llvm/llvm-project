# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .data
# CHECK: TEST0:
# CHECK: .byte 0
TEST0:
        .base64 "AA=="

# CHECK: TEST1:
# CHECK: .ascii "abcxyz"
TEST1:
        .base64 "YWJjeHl6"
