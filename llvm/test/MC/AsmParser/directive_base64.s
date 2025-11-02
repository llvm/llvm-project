# RUN: not llvm-mc -triple i386-unknown-unknown %s | FileCheck %s
# RUN: not llvm-mc -triple i386-unknown-unknown -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

        .data
# CHECK: TEST0:
# CHECK: .byte 0
TEST0:
        .base64 "AA=="

# CHECK: TEST1:
# CHECK: .ascii "abcxyz"
TEST1:
        .base64 "YWJjeHl6"

# CHECK: TEST2:
# CHECK-ERROR: error: expected string
TEST2:
        .base64 not-a-string

# CHECK: TEST3:
# CHECK-ERROR: error: failed to base64 decode string data
TEST3:
        .base64 "AA"

# CHECK: TEST4:
# CHECK-ERROR: error: expected nonempty string
TEST4:
        .base64 ""
