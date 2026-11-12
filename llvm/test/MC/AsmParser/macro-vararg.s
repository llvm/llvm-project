# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s
.macro one a:vararg
.ascii "|\a"
.endm

# CHECK:     .byte 124
one
# CHECK:     .ascii "|1"
one 1
## Difference: GNU as squeezes repeated spaces.
# CHECK:     .ascii "|1  2"
one 1  2
## Difference: GNU as non-x86 drops the space before '(' (gas PR/25750)
# CHECK:     .ascii "|1  (2  3"
one 1  (2  3
# CHECK:     .ascii "|1  2  3)"
one 1  2  3)

.macro two a, b:vararg
.ascii "|\a|\b"
.endm

# CHECK:     .ascii "||"
two
# CHECK:     .ascii "|1|"
two 1
## Difference: GNU as squeezes repeated spaces.
# CHECK:     .ascii "|1|2  3"
two 1 2  3

## Parameters can be separated by spaces
.macro two1 a b:vararg
.ascii "|\a|\b"
.endm

# CHECK:     .ascii "|1|2"
two1 1  2
