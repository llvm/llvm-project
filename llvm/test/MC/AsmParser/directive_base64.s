# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s
# RUN: not llvm-mc -triple i386-unknown-unknown -defsym=ERR=1 -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

        .data
# CHECK-LABEL: TEST0:
# CHECK-NEXT: .byte 0
TEST0:
        .base64 "AA=="

# CHECK-LABEL: TEST1:
# CHECK-NEXT: .ascii "abcxyz"
TEST1:
        .base64 "YWJjeHl6"

# CHECK-LABEL: TEST2:
# CHECK-NEXT: .byte 1
# CHECK-NEXT: .byte 2
TEST2:
        .base64 "AQ=="
        .base64 "Ag=="

# CHECK-LABEL: TEST3:
# CHECK-NEXT: .byte 1
# CHECK-NEXT: .byte 2
TEST3:
        .base64 "AQ==", "Ag=="

.ifdef ERR
# CHECK-ERROR: [[#@LINE+1]]:17: error: expected string
        .base64 not-a-string

# CHECK-ERROR: [[#@LINE+1]]:17: error: failed to base64 decode string data
        .base64 "AA"

# CHECK-ERROR: [[#@LINE+1]]:17: error: expected nonempty string
        .base64 ""
.endif
